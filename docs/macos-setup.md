# macOS setup

The reference environment in `docs/phase-log.md` is an NVIDIA machine. `requirements.lock`
cannot install on macOS: it pins `cuda-toolkit`, `cuda-bindings` and a CUDA build of torch,
none of which ship Apple Silicon wheels. This file covers the Apple Silicon environment
instead.

## Machine

| | |
| --- | --- |
| Hardware | Apple Silicon (arm64) |
| OS | macOS |
| Python | 3.13.5 (system; the reference uses 3.12) |
| uv | 0.11.30 |
| Accelerator | MPS (torch 2.15.0, `mps: True`) |

## The blocker

```bash
uv pip install -r requirements.lock   # fails: no cuda-* wheels for darwin/arm64
```

`requirements.lock` is correct for the PC and stays that way. It only applies to the machine
that produced it.

## Solution: one lockfile per platform

`requirements.txt` stays the single source of truth. Each platform compiles it.

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip compile requirements.txt -o requirements-macos.lock
uv pip install -r requirements-macos.lock
```

That resolves 26 packages and takes about a minute to download and install.

## Verify

```bash
python -c "import numpy, scipy, pandas, pyarrow, joblib, pytest, torch, brian2; print('all ok')"
python -c "import torch; print(torch.__version__, 'mps:', torch.backends.mps.is_available())"
python -c "import brian2; print('brian2', brian2.__version__)"
```

```
all ok
2.15.0 mps: True
brian2 2.11.0
```

## Data

Two files from the FlyWire v783 Zenodo record (10.5281/zenodo.10676866), about 853 MB total.

```bash
curl -L --fail -o data/raw/proofread_connections_783.feather \
  "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"

curl -L --fail -o data/raw/proofread_root_ids_783.npy \
  "https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy?download=1"
```

| File | Size | md5 |
| --- | --- | --- |
| `proofread_connections_783.feather` | 852 MB | `f48f972d262323a102aed49af1396b8a` |
| `proofread_root_ids_783.npy` | 1.1 MB | `e0e6c19732fd8c7a4e39a2d170105421` |

Both md5s match the Zenodo record. Download speed varied a lot: it started around 270 KB/s,
and the completed transfer averaged about 1.4 MB/s over 9 minutes. Worth asking around for a
local copy before starting.

## Tests

This machine keeps the reference clone at `~/Desktop/Drosophila_brain_model`, so the
comparison test needs `SHIU_DIR`. The default is `~/Downloads/Drosophila_brain_model`.

```bash
# without the reference clone
pytest -q          # 1 passed, 1 skipped, 8.84 s

# with it
SHIU_DIR=~/Desktop/Drosophila_brain_model pytest -q
# 2 passed, 16.18 s
```

The second test loads Shiu's `Connectivity_783.parquet` (100 MB) and checks that the edge set
and synapse counts match ours: 15,091,983 edges, 54,492,922 total weight.

## What went wrong

| Problem                                          | Cause                                                               | Fix                                                     |
|:-------------------------------------------------|:--------------------------------------------------------------------|:--------------------------------------------------------|
| `requirements.lock` won't install                | `cuda-*` packages have no macOS wheels                              | Separate `requirements-macos.lock`                      |
| System Python is 3.13, reference says 3.12       | Different machine                                                   | 3.13 has working wheels for every dependency            |
| `pytest: command not found` in a second terminal | New shell, venv not activated                                       | `source .venv/bin/activate`, or call `.venv/bin/pytest` |
| Tests "pass" while testing nothing               | Both tests `skipif` when the data or the reference clone is missing | A "2 skipped" run means nothing ran                     |
| Brian2 resolves to 2.11.0, reference pins 2.10.1 | Installed from `requirements.txt`, not the lock                     | Accepted for now; pin if reference-run parity is needed |
| Comparison test skipped                          | Clone is on the Desktop, the test defaults to `~/Downloads`         | `SHIU_DIR=<path>`                                       |
| `git clone` into a pre-existing folder fails     | Destination must be empty                                           | Let `git clone` create the directory                    |

## Differences from the reference environment

| Component   | PC (reference)                | macOS                               |
|:------------|:------------------------------|:------------------------------------|
| Python      | 3.12                          | 3.13.5                              |
| Lockfile    | `requirements.lock` (45 pins) | `requirements-macos.lock` (26 pins) |
| torch       | 2.15.0 + cu132                | 2.15.0, CPU/MPS                     |
| brian2      | 2.10.1                        | 2.11.0                              |
| pandas      | 3.1.0                         | 3.1.1                               |
| Accelerator | CUDA (RTX 3050, 4 GB)         | MPS                                 |

Benchmark numbers from the two machines cannot be compared. Phase 1's CPU-vs-GPU gate has to
be measured on each machine separately.

## Notes for the next person

- Peak RAM during a full load is about 4.7 GB: pandas holds the 16.8M-row table before the
  CSR matrix is built. It drops back afterwards, and nothing is leaking.
- `data/raw/*`, `*.npy` and `*.npz` are git-ignored. The 853 MB download is never committed.
- The Brian2 reference runner has not been re-run on this machine. When it is, check the
  brian2 version: 3.2 GB per worker on 2.10.1 is the documented figure, and 2.11.0 is
  untested here.
- The full test run takes about 9 s here and produces a 121 MB CSR matrix, well under the
  2.5 GB Phase 1 memory gate.