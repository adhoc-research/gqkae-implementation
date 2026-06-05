# Future expansion outside the MVP

The approved MVP intentionally targets only H4 `(4e,4o)` with 6-31G, bond length `0.88 Å`, and circuit length `L=20`.

Future work to reproduce the full paper should add:

- Potential-energy scans for H4 and all other reported systems.
- N2, LiH, C2H6, H2O, and H2O dimer active-space setup/configs.
- Publication-style convergence/error/PES plots.
- Exact CUDA-Q decompositions for each fermionic excitation operator and backend-specific gate-count extraction.
- Multi-seed aggregation and paper table generation.
- FlashQKAN/cuQuantum acceleration for the QKAN latent processor.
