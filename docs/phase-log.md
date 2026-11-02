# Fly-Society Phase Log

Engineering log for each phase in `dev-plan.md`: what was built, what it measured, what broke, what is left. Unfamiliar term? See the [Glossary](#glossary) at the end.

## The project in one paragraph

A fruit fly's brain has been fully mapped as a **directed weighted graph**: 139,255 nodes (neurons) and ~15M edges (connections). We run that graph as a program. Each node holds one number (its "voltage"). Every tick, voltages decay a bit, each node adds up the weights of incoming edges from nodes that just fired, and any node over a threshold fires and resets. For all nodes at once, that is a sparse matrix-vector multiply, `W @ spikes`. Later phases attach this "brain" to a body in a 2D world, then run many copies side by side and look for group behaviour.

| Phase | What it is in CS terms | Status |
| --- | --- | --- |
| 1 | Load the graph, validate against a reference implementation, benchmark one instance | In progress |
| 2 | Wire world input into some nodes and read movement out of others; one agent in a 2D world | Not started |
| 3 | Prune the graph to the subgraph that matters, for speed | Not started |
| 4 | Batch 2 agents over one shared read-only graph; prove no state leaks between them | Not started |
| 5 | Scale to 10–100 agents; custom kernels, spatial hashing, async logging | Not started |
| 6 | Run experiments and compare against randomized-graph controls | Not started |

---

## Phase 1: Load, validate, benchmark

**Goal:** fit the whole graph on a 16 GB RAM / RTX 3050 (4 GB VRAM) laptop, prove the simulation gives the same answer as a published implementation, and measure ms per tick on CPU vs GPU.

**Dates:** 2026-11-01 to 2026-11-02 (so far)

| Task | Status |
| --- | --- |
| Download the graph data | Done |
| Build the sparse adjacency matrix | Done |
| Run the reference implementation and check its known output | Done |
| Benchmark: 1,000 ticks on CPU vs GPU | Not started |
| Determinism: same seed gives the same output | Not started |

### 1. Environment

- Python **3.12** venv via `uv`. System Python is 3.14, too new for some torch/brian2 wheels.
- `requirements.txt`: top-level deps (numpy, scipy, pandas, pyarrow, joblib, torch, brian2, pytest).
- `requirements.lock`: all 43 packages pinned (torch 2.15.0+cu132, brian2 2.10.1, numpy 2.5.4, scipy 1.18.1).
- torch sees the GPU. The CUDA runtime ships inside the torch wheel, so only the NVIDIA driver (615.78) is needed on the system.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
```

### 2. Data

Source: FlyWire v783 release on Zenodo. FlyWire is the project that mapped the brain; v783 is the current public version. Files saved in `data/raw/` (git-ignored), md5-verified:

| File | Size | Contents |
| --- | --- | --- |
| `proofread_connections_783.feather` | 852 MB | Edge list, 16.8M rows: `pre_id, post_id, region, syn_count, 6 chemical-type probabilities` |
| `proofread_root_ids_783.npy` | 1.1 MB | The 139,255 node IDs (18-digit ints) |

Skipped: a 9.5 GB file of per-connection 3D coordinates (not needed), and per-region count tables (maybe useful in Phase 3).

Kept the raw files instead of the reference lab's preprocessed copy. The raw data keeps the `region` column (Phase 3 and 6 need it) and the raw probabilities (so we can change the sign rule later).

We also cloned the reference implementation (Shiu et al., `philshiu/Drosophila_brain_model`) to `~/Downloads/Drosophila_brain_model`. It has its own preprocessed copy of the same graph, which we use for cross-checking.

### 3. Sparse adjacency matrix: `core/connectome.py`

`load_connectome()` returns `(W, ids)`. `W` is a 139,255 × 139,255 `float32` CSR matrix.

- **Why sparse:** dense would be 19.4B cells, about 77 GB. Only 15M cells are non-zero, so CSR takes **121 MB**.
- **Layout:** `W[target, source]`, rows are targets. Each node's input is then one contiguous row dot product, and the batched version for many agents (`W @ V`, `V` is `[N, B]`) reads memory the same way.
- **Weight:** number of physical connections between the pair (`syn_count`), summed over all rows for that pair (one row per brain region).
- **Sign:** an edge is negative when the source node inhibits ("calm down") rather than excites ("fire"). The data gives, per edge, probabilities over 6 chemical types the source uses. For each source node, sum those probabilities weighted by `syn_count` and take the argmax. The two inhibitory chemicals (GABA, glutamate) give −1, everything else +1. Same convention as the reference.
- **Dedup trick:** building a COO matrix and calling `.tocsr()` sums duplicate `(target, source)` entries, so no pandas groupby is needed.
- **Input check:** raises if an edge references an ID that isn't in the node list.

Tests (`tests/test_connectome.py`, `.venv/bin/pytest -q`, ~8 s):

- shape, dtypes (`float32` values, `int32` indices), 15,091,983 edges, 54,492,922 total weight
- edges and weights **identical** to the reference lab's copy
- signs agree on more than 95% of edges
- tests skip, not fail, when the data or the reference repo isn't on disk

### 4. Reference implementation as an integration test

The reference paper has a known input → expected output pair:

- **Input:** force 21 specific nodes (sugar-taste sensors) to fire randomly at 150 Hz. Think "the fly tastes sugar."
- **Expected output:** a node called **MN9** starts firing. MN9 drives the fly's feeding reflex (sticking its mouthpart out).

We ran their code unchanged (it uses the Brian2 simulator) on the graph version their paper used (v630): 30 runs × 1 s simulated, 2 worker processes. Script: `experiments/shiu_sugar_v630.py`.

```bash
cd ~/Downloads/Drosophila_brain_model
PYTHONPATH=. ~/Downloads/fly-society/.venv/bin/python ~/Downloads/fly-society/experiments/shiu_sugar_v630.py 30 2 sugarR
```

Output: `~/Downloads/Drosophila_brain_model/results/example/sugarR.parquet`, one row per spike.

### Results

**Matrix (v783)**

| Metric | Value |
| --- | --- |
| Nodes | 139,255 |
| Edges | 15,091,983 |
| Total weight | 54,492,922 |
| Memory | 121 MB (plan budget 450 MB) |
| Load time | 5.5 s |
| Peak RAM while building | 4.7 GB (temporary pandas table) |
| Negative edges | 40.2% |
| Sign agreement with reference | 95.7% of source nodes |

**Integration test (v630)**

| Check | Expected | Got |
| --- | --- | --- |
| Total fire events, 30 runs | > 400,000 | 408,165 |
| Nodes that ever fired | ~400 | 437 |
| MN9 fires | yes | 83.6 per second (rank 36 of 437) |
| The 21 forced nodes | ~150 per second | 141–153 per second |
| Wall time | n/a | 698 s on 2 workers |
| RAM per worker | n/a | ~3.2 GB |

**Verdict:** the reference reproduces its published result on this machine. The matrix is far under the 2.5 GB memory target; per-agent state still has to be added and measured.

### What went wrong

| Problem | Cause | Fix / status |
| --- | --- | --- |
| Python 3.14 unsuitable | Too new for some wheels | 3.12 venv |
| I assumed the reference dropped weak edges (below some minimum count) | Assumption, not checked | The comparison showed an exact match with no filter. The test now checks this directly. |
| 1,732 edge rows have no chemical-type data (all NaN) | Gaps in the source data | Treated as 0. Every source node still gets a sign from its other edges. |
| Signs match the reference on only 95.7% | The reference used a separate per-node lookup table instead of deriving signs from edges | Accepted; ~5,900 nodes differ. Use their sign column if exact reproduction is needed. |
| Test failed on total weight | Summing 15M `float32` values loses precision past 2²⁴ ≈ 16.7M | Sum in `float64`. The weights themselves are exact small ints. Rule: global sums use float64. |
| Reference code failed to import | Missing `joblib` | Added to requirements |
| Runner script couldn't `import model` | Python adds the script's folder to `sys.path`, not the working directory | Run with `PYTHONPATH=.` |
| Reference uses ~3.2 GB per process | Brian2 stores every edge as an object with its own state and event queue | Ran 2 workers to fit RAM. This is why we write our own stepper on the 121 MB CSR (26× smaller). |
| Loader peaks at 4.7 GB | Whole 16.8M-row table in pandas | Open. Cache `W` once with `scipy.sparse.save_npz` to `data/processed/`. |

### Gaps

- The integration test ran on graph **v630**; our loader uses **v783**. Node IDs differ between versions, so the 21 input nodes and MN9 need their v783 IDs looked up before our stepper can be checked.
- Compared only the headline numbers and MN9, not per-node rates from the paper's data archive.
- No CPU vs GPU benchmark yet.
- No determinism test yet.

### Next

1. **Our own stepper in torch** on `W`. The reference's per-tick update with its constants:
   - decay toward −52 (time constant 20 ms); fire above −45; reset to −52
   - incoming input decays with time constant 5 ms; each unit of edge weight adds 0.275
   - after firing, a node is locked out for 2.2 ms
   - a fire event reaches its targets after 1.8 ms. At 0.1 ms ticks that is 18 ticks, so keep a ring buffer of the last 18 spike vectors.
2. Rerun the sugar → MN9 test with it on v783 and compare against the reference within run-to-run noise.
3. Benchmark 1,000 ticks on CPU and GPU: ms per tick, peak RAM, peak VRAM.
4. Determinism test: same seed, identical output.
5. Optional: `.npz` cache for `W`.

### Files added

| Path | Purpose |
| --- | --- |
| `requirements.txt`, `requirements.lock` | Environment |
| `core/connectome.py` | Sparse matrix loader |
| `tests/test_connectome.py`, `pytest.ini` | Loader tests, import path |
| `experiments/shiu_sugar_v630.py` | Reference integration-test runner |
| `docs/phase-log.md` | This log |

---

## Phase 2: One agent in a 2D world

Not started.

## Phase 3: Graph pruning

Not started.

## Phase 4: Two agents, shared graph

Not started.

## Phase 5: Scale to 10–100 agents

Not started.

## Phase 6: Experiments and controls

Not started.

---

## Glossary

Each term with the CS idea it maps to. Add new terms here as phases introduce them.

### Biology terms

| Term | CS equivalent | Notes |
| --- | --- | --- |
| Neuron | Graph node with mutable state | State is one float ("voltage") plus a lockout timer. |
| Synapse | One physical link between two nodes | Many synapses between the same pair add up to one weighted edge. |
| Connectome | The full graph: nodes + weighted directed edges | For the fly: 139,255 nodes, 15,091,983 edges. |
| Spike / fire | Boolean event emitted when a node's state crosses a threshold | Node resets after firing. The output of the whole system is a stream of these. |
| Firing rate (Hz) | Events per second for one node | "MN9 at 83.6 Hz" = it fired 83.6 times per simulated second. |
| Voltage (membrane potential) | The node's state variable | Units are millivolts, but treat it as a float that rises with input and decays toward rest. |
| Excitatory / inhibitory | Positive / negative edge weight | Excitatory pushes the target toward firing; inhibitory pushes it away. |
| Neurotransmitter | Chemical a node sends over its edges | Only used here to decide the edge sign. GABA and glutamate mean negative. |
| Refractory period | Cooldown after firing | 2.2 ms during which the node cannot fire again. |
| Synaptic delay | Message latency on an edge | 1.8 ms between a fire event and its effect on targets. |
| Neuropil | Named brain region | Like a module or namespace. The raw data has one edge row per region. |
| Sensory neuron | Input node | Driven by the outside world (taste, smell, touch). Phase 2 writes world data into these. |
| Motor / descending neuron | Output node | Drives the body. Phase 2 reads movement from these. |
| Sugar neurons (GRNs) | The 21 input nodes used in the integration test | Taste sensors that respond to sugar. |
| MN9 | The output node checked in the integration test | Drives the feeding reflex (fly sticks its mouthpart out). |
| Lesion | Deleting a node or zeroing its edges | Used in Phase 6 to test which parts of the graph cause a behaviour. |
| *Drosophila melanogaster* | The fruit fly | The species whose brain was mapped. |

### Data and tools

| Term | Meaning |
| --- | --- |
| FlyWire | The project that mapped the fly brain and publishes the graph data. |
| v630 / v783 | Data versions. v630 was used by the reference paper; v783 is current and used by our loader. Node IDs differ between versions. |
| Root ID | FlyWire's 18-digit node ID, e.g. `720575940660219265`. |
| Proofread | Nodes that humans checked by hand. v783 has 139,255. |
| Shiu et al. | The reference paper and repo (`philshiu/Drosophila_brain_model`) that first simulated this whole graph. Our integration-test baseline. |
| Brian2 | The Python simulator the reference uses. Simple to use but heavy: ~3.2 GB per process for this graph. |
| Feather / Parquet | Columnar binary table formats. Much faster to load than CSV. |

### Model and math

| Term | Meaning |
| --- | --- |
| LIF (Leaky Integrate-and-Fire) | The per-tick update rule: state decays toward rest (leaky), adds incoming weights (integrate), fires and resets past a threshold (fire). |
| Tick / Δt | One simulation step, 0.1 ms of simulated time. 1 simulated second = 10,000 ticks. |
| CSR (Compressed Sparse Row) | Sparse matrix stored as 3 arrays: `data` (values), `indices` (column of each value), `indptr` (where each row starts). Stores only non-zero cells. |
| COO | Sparse matrix as `(row, col, value)` triples. Easy to build. Converting to CSR sums duplicate entries. |
| SpMV / SpMM | Sparse matrix × dense vector / dense matrix. `W @ spikes` for one agent (SpMV); `W @ V` with `V` of shape `[N, B]` for B agents (SpMM). |
| Poisson input | Random events at a fixed average rate, each tick independent. Used to force input nodes to fire. |
| Ring buffer | Fixed-size circular array. Holds the last 18 spike vectors to implement the 1.8 ms edge delay. |
| Determinism | Same seed and inputs give bit-identical output. |
| Shuffled-weights control | The same graph with edge weights randomly reassigned. A baseline: if a behaviour survives shuffling, the real wiring isn't what causes it. |
