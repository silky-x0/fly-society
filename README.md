```text
              _  _
             ( \/ )
      .---.   \  /   .---.
     /   _ \.-'  '-./ _   \       F L Y - S O C I E T Y
    |   ( \  / () \  / )   |      ---------------------
     \   \ \/      \/ /   /       Multi-Agent Artificial Life Powered by
      '-._\  /\  /\  /_.-'        Biological Connectome Architecture
         / /  |  |  \ \
        /_/   |__|   \_\          [ ~139k Neurons  |  ~50.7M Synapses ]
       /      /  \      \         [ Shared CSR     |  Batched SpMM    ]
             '    '
```

# Fly-Society

An experimental artificial life simulator exploring emergent collective behavior using biological neural substrates.

### Why This Exists
I have always loved psychology; the fundamental question of *why people (and living things) do what they do*. How do basic internal drives, sensory inputs, and physical wiring combine to create attraction, avoidance, competition, and complex social dynamics? 

So rather than studying this purely through abstract behavioral theory, here I am attempting to build this simulation from the ground up: giving virtual organisms genuine biological brains mapped from real neurobiology, placing them in a shared world, and observing what behaviors naturally emerge when they interact.

---

### What It Is
Instead of relying on hardcoded heuristics (like Boids) or black-box reinforcement learning, every agent in this environment is driven by neural circuitry derived from the mapped **Drosophila melanogaster (FlyWire) connectome**. Agents share a single immutable synaptic connectivity graph in memory while maintaining strictly isolated membrane potentials, sensory histories, and motor states.

### Core Engineering Thesis
Can biological wiring alone—when placed inside a shared 2D physics arena with chemical gradients and physical obstacles—produce measurable social dynamics like resource competition, clustering, and avoidance without explicit social programming?

### Architecture Highlights
* **Zero-Copy Static Topology:** The ~50.7M synapse connectome is compressed into a read-only `float32` CSR matrix shared across all living instances.
* **Batched State Stepping:** Agent neural states ($V, R, I$) are evaluated concurrently via Sparse-Dense Matrix Multiplication (`SpMM`).
* **Sensory-Motor Transduction:** Environmental stimuli (odor plumes, raycast collisions) are Poisson-encoded directly into biological sensory root IDs, while Descending Neurons (DNs) are decoded into 2D kinematics.

---

### Development Status

- [ ] **Phase 1: Baseline Benchmarking & Connectome Ingestion** *(In Progress)*
- [ ] **Phase 2: Sensory-Motor Transduction & Single-Agent Embodiment**
- [ ] **Phase 3: Pathway Pruning & Sub-Circuit Extraction**
- [ ] **Phase 4: Dyadic Interaction & State Isolation Audit**
- [ ] **Phase 5: Population Scaling (10–100 Agents) & C++/CUDA Kernels**
- [ ] **Phase 6: Collective Emergence & Lesion Experiments**
