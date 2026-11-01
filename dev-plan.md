This execution blueprint translates your architectural concept into a phased engineering roadmap. Each phase isolates a specific technical risk—starting with raw hardware feasibility and ending with multi-agent emergence—so you never build infrastructure on top of unverified assumptions.

---

### System Architecture & Memory Layout

Before starting Phase 1, anchor your codebase around this strict separation of static topology and dynamic agent state:

```
+-----------------------------------------------------------------------+
|                   SHARED IMMUTABLE MEMORY (Read-Only)                 |
|  FlyWire Connectome CSR Matrix: indptr [N+1], indices [M], weights [M]|
+-----------------------------------------------------------------------+
                                   |
          +------------------------+------------------------+
          |                                                 |
          v                                                 v
+----------------------------------+      +----------------------------------+
|       AGENT 1 STATE (R/W)        |      |       AGENT N STATE (R/W)        |
|  - Membrane Potential: V[N]      |      |  - Membrane Potential: V[N]      |
|  - Refractory Timer:   R[N]      |      |  - Refractory Timer:   R[N]      |
|  - Synaptic Current:   I[N]      |      |  - Synaptic Current:   I[N]      |
|  - 2D Pose:            (x, y, θ) |      |  - 2D Pose:            (x, y, θ) |
+----------------------------------+      +----------------------------------+
          ^                  |                      ^                  |
   Sensory|                  |Motor          Sensory|                  |Motor
   Spikes |                  |Vectors        Spikes |                  |Vectors
          |                  v                      |                  v
+-----------------------------------------------------------------------+
|                     2D WORLD ENGINE & SPATIAL GRID                    |
|  - Chemical Plume Diffusion Grid      - Raycast Vision / Collisions   |
|  - SQLite Telemetry Logger            - WebSocket State Broadcaster   |
+-----------------------------------------------------------------------+

```

---

## Phase 1: Baseline Benchmarking & Connectome Ingestion

**Objective:** Run a single, disembodied whole-brain simulation (~139k neurons, ~50.7M synapses) locally to establish hardware limits on your 16 GB RAM / RTX 3050 (4 GB VRAM) setup.

#### Engineering To-Do List

* [ ] **Ingest Connectome Data:** Download the FlyWire synaptic connectivity dataset and neuron classification tables (sensory, ascending, descending, motor).
* [ ] **Build CSR Matrix Loader:** Convert the raw edge list into a SciPy/NumPy Compressed Sparse Row (`CSR`) matrix using `float32` for weights and `int32` for indices to keep the static graph footprint under ~450 MB.
* [ ] **Reproduce Reference Model:** Clone the Shiu et al. (Brian2/PyTorch) or *Fruit Fly Laboratory* repository and run a baseline sensory-stimulation experiment (e.g., sugar-sensing gustatory receptor activation).
* [ ] **Profile Compute Backends:** Run 1,000 simulation ticks (at $\Delta t = 0.1\text{ ms}$) on both CPU (multi-threaded) and GPU (CUDA). Log wall-clock execution time, peak RAM, peak VRAM, and L3 cache miss rates.
* [ ] **Verify Determinism:** Fix random seeds for Poisson spike generators and verify that two consecutive runs produce identical spike rasters.

#### Success Criteria (Exit Gate)

* **Memory Gate:** The static connectome + 1 neural state vector fits comfortably in under **2.5 GB of system RAM** (or **1.5 GB VRAM**).
* **Validation Gate:** Stimulating known gustatory receptor neurons (GRNs) reliably triggers firing in the expected downstream motor neuron clusters (e.g., proboscis extension reflex circuit) matching published reference data.
* **Metric Gate:** You have a hard numbers table showing exact `ms per simulated tick` on CPU vs. GPU.

---

## Phase 2: The Transduction Layer & Single-Agent Embodiment

**Objective:** Close the sensory-motor loop by placing one connectome-driven agent inside a 2D physics world and proving it can react to environmental stimuli.

#### Engineering To-Do List

* [ ] **Build 2D World Engine:** Write a lightweight, headless 2D continuous-space physics engine in Python/C++ supporting circular rigid bodies, obstacle boundaries, and radial chemical gradients (food plumes).
* [ ] **Map Sensory Receptors (Input Adapter):** Identify specific FlyWire root IDs for olfactory receptor neurons (ORNs), gustatory neurons, and mechanosensory bristles.
* [ ] **Implement Poisson Rate Coding:** Write a transduction function that converts local 2D chemical concentration $C(x, y)$ and bilateral antenna positions $(x_L, y_L), (x_R, y_R)$ into Poisson firing rates ($\text{Hz}$) injected directly into the corresponding ORN indices.
* [ ] **Map Descending Neurons (Output Adapter):** Identify key Descending Neurons (DNs) that project to the ventral nerve cord (VNC) controlling forward walking, halting, and left/right turning.
* [ ] **Implement Motor Decoder:** Create a sliding-window firing-rate integrator (e.g., 20 ms window) over left vs. right turning DNs that translates spike differentials into angular velocity ($\omega$) and forward thrust ($v$).
* [ ] **Build Debug Visualizer:** Set up a basic React + Three.js (or HTML5 Canvas) frontend receiving state snapshots over WebSockets to visually inspect the fly's trajectory and live firing rates of key motor neurons.

#### Success Criteria (Exit Gate)

* **Chemotaxis Gate:** Placed near a virtual odor gradient, a single embodied agent reliably orientates toward and reaches the food source significantly faster than a random-walk baseline.
* **Obstacle Avoidance Gate:** Mechanosensory input triggered by wall collisions induces turning behavior that frees the agent from corners.

---

## Phase 3: Pathway Pruning & Circuit Reduction

**Objective:** Strip away neural sub-circuits irrelevant to 2D foraging and social interaction (e.g., flight stabilization, halteres, deep optic lobe layers) to achieve a 5x–10x speedup without breaking Phase 2 behaviors.

#### Engineering To-Do List

* [ ] **Graph Traversal & Subgraph Extraction:** Using the Phase 2 sensory inputs as sources and motor DNs as sinks, run a $k$-hop shortest-path / flow traversal across the FlyWire graph to isolate the active sensorimotor and central complex (navigation) pathways.
* [ ] **Prune Dormant Sub-networks:** Remove isolated or non-contributing subgraphs (such as high-resolution lamina/medulla visual processing if using simplified raycast vision) and re-index the CSR matrix.
* [ ] **Recalibrate Firing Thresholds:** Removing background synaptic input changes resting potentials; tune baseline leak conductances or tonic background input so the reduced circuit maintains biological firing rates.
* [ ] **Build Automated Regression Suite:** Run the Phase 2 chemotaxis and obstacle tests on both the Full Brain (Level 1) and Reduced Circuit (Level 2) across 20 trials.

#### Success Criteria (Exit Gate)

* **Fidelity Gate:** The reduced circuit achieves **at least 85% behavioral correlation** with the full-brain model on food-finding latency and trajectory smoothness.
* **Performance Gate:** Per-agent compute time and state memory footprint drop by at least **60–80%**, unlocking multi-agent scaling.

---

## Phase 4: Multi-Agent Architecture & Dyadic Interaction

**Objective:** Run two independent neural states sharing a single immutable connectome in RAM, and observe genuine agent-to-agent interaction in a shared arena.

#### Engineering To-Do List

* [ ] **Batched State Matrix Architecture:** Refactor the neural stepper so connectivity remains a `[N, N]` sparse matrix while neural states ($V, R, I$) become dense `[N, B]` matrices (where $B = \text{population size}$), replacing single-vector ops with Sparse-Dense Matrix Multiplication (`SpMM`).
* [ ] **Memory Isolation Audit:** Write a unit test that clamps Agent 1's sensory inputs to zero while blasting Agent 2 with maximum sensory input; verify zero floating-point drift or cross-talk in Agent 1's state arrays.
* [ ] **Implement Inter-Agent Sensory Cues:** Give each agent a short-range pheromone aura and physical collision radius so Agent A's proximity triggers mechanosensory and conspecific olfactory/visual pathways in Agent B.
* [ ] **Implement Null-Model Control Modes:** Add a configuration flag to run agents with **Shuffled Weights** (same graph degree distribution, randomized synapse pairings) to serve as a scientific control group.

#### Success Criteria (Exit Gate)

* **Zero Cross-Talk Gate:** Unit tests confirm 100% state isolation between agents sharing the static CSR graph.
* **Dyadic Interaction Gate:** Two active agents placed near a single small food source exhibit measurable non-random interaction (e.g., displacement, chasing, or prolonged proximity) that statistically diverges from the Shuffled-Weight control group.

---

## Phase 5: Population Scaling (10 to 100 Agents) & Kernel Optimization

**Objective:** Push the simulator to 10–100 concurrent agents within your 16 GB RAM ceiling by moving hot loops into compiled C++/CUDA kernels and implementing decoupled telemetry.

#### Engineering To-Do List

* [ ] **Custom C++ / CUDA `SpMM` Kernel:** Profile the batched stepper; rewrite the bottleneck synaptic propagation and leaky integrate-and-fire (LIF) state update step in C++ (with OpenMP/SIMD) or a custom PyTorch/CUDA C++ extension.
* [ ] **Spatial Hashing Grid:** Upgrade the 2D physics engine from $O(B^2)$ pairwise distance checks to an $O(B)$ spatial hash grid for fast neighbor lookups and sensory raycasting.
* [ ] **Asynchronous SQLite Telemetry Pipeline:** Decouple data logging from the main simulation loop using a background worker thread that batch-inserts agent positions, spike summaries, and interaction events into SQLite every $K$ ticks.
* [ ] **Frontend Replay & Live Dashboard:** Upgrade the React + Three.js viewer to render up to 100 instanced meshes (`THREE.InstancedMesh`) with playback controls (play, pause, scrub SQLite history) and live charts for population clustering.

#### Success Criteria (Exit Gate)

* **Scale & Memory Gate:** A 50-to-100 agent simulation runs stably for 100,000+ ticks without exceeding **12 GB total system RAM** or triggering VRAM out-of-memory crashes.
* **Throughput Gate:** Simulation achieves a practical throughput benchmark (e.g., $\ge 10\text{x}$ faster than single-threaded sequential execution), allowing a 60-second biological experiment to finish in minutes rather than days.

---

## Phase 6: Collective Behavior Experiments & Analysis

**Objective:** Run structured artificial-life experiments across population densities and resource distributions to prove emergent social dynamics driven by biological wiring.

#### Engineering To-Do List

* [ ] **Experiment 1 (Resource Patch Competition):** Run populations of $B \in \{1, 10, 25, 50, 100\}$ with 1 vs. 4 food patches. Measure Gini coefficients of food consumption and displacement frequency.
* [ ] **Experiment 2 (Density-Dependent Clustering):** Run agents in a uniform, food-free arena across increasing densities. Compute spatial pair-correlation functions and cluster lifetimes.
* [ ] **Experiment 3 (Lesion / Ablation Studies):** Digitally "lesion" specific social or olfactory pathways (zeroing out specific rows in the shared CSR matrix) and observe how collective clustering or competition collapses.
* [ ] **Data Analysis Pipeline:** Write Python (`pandas` / `scipy.stats`) scripts to query the SQLite logs and generate comparative plots: **Biological Connectome** vs. **Shuffled-Weight Connectome** vs. **Boids/Rule-Based Baseline**.

#### Success Criteria (Exit Gate)

* **Scientific Emergence Gate:** Statistical tests ($p < 0.01$) demonstrate that FlyWire-driven populations produce structured collective metrics (clustering, resource partitioning, or avoidance cascades) that vanish when synaptic weights are shuffled or when specific biological pathways are lesioned.

---

### Phase Summary & Resource Allocation Matrix

| Phase | Target Population | Circuit Scope | Primary Bottleneck | Key Deliverable |
| --- | --- | --- | --- | --- |
| **1. Baseline** | 1 (Disembodied) | Full (~139k neurons) | RAM Bandwidth | Hardware benchmark & CSR loader |
| **2. Embodiment** | 1 (2D Body) | Full (~139k neurons) | Transduction Math | Working sensory-motor closed loop |
| **3. Pruning** | 1 (2D Body) | Reduced (Sub-circuit) | Biological Validity | Pruned CSR graph + regression suite |
| **4. Dyadic** | 2 (Shared Arena) | Reduced (Sub-circuit) | State Isolation | Batched `SpMM` stepper + control group |
| **5. Scaling** | 10 – 100 | Reduced (Sub-circuit) | Compute / Cache | C++/CUDA kernels + Three.js dashboard |
| **6. Emergence** | 10 – 100 | Reduced + Lesions | Statistical Rigor | Comparative emergence dataset in SQLite |