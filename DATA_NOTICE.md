# Data Provenance and Scope Notice (Paper 21)

This repository contains the standalone reproducibility package for **Paper 21** (`tau-core-black-hole-cosmology-duality-paper21`):
*Black Holes, the Big Bang, and Present Cosmic Expansion as Dual Painlevé–Gullstrand Regimes of the Discrete $K_4$ Temporal Substrate: Non-Singular Anatomy, Horizon Thermodynamics, and Multi-Probe Validation ($k = 0$)*.

## Included Public & Upstream Frozen Datasets

### 1. Standalone Vendored Upstream Summaries (`data/raw/`)
- `euclid_dr1_live_validation_results.json`: Source-frozen summary of `18,983` live observational records queried from the public **ESA Euclid DR1-Foundation (2026) Science Archive** (`eas.esac.esa.int/tap-server/tap/sync`, schema `dr1.*`), covering `5,955` unique 3D-mapped extragalactic objects (`359` NISP spectroscopic galaxies across $0.1309 \le z_{\mathrm{spec}} \le 3.4927$, `5,601` unique photometric/morphological galaxies including `524` massive foreground lenses, and `354` QSO cosmic-web tracers), including the `2,859` positive-clock void objects peaking at $+484.04\;\mathrm{ppm}$ and deepest cluster clock wells at $-289.60\;\mathrm{ppm}$ (imported from Paper 20).
- `frontier_three_pillars_results.json`: Source-frozen Three-Pillar mathematical and high-$z$ empirical derivation ledger (`Theorems F1–F4`, including the `43` JWST+HST `Mancera Piña+ (2025)` disks at $z \approx 0.90$, `3` Cosmic Noon ALMA cold disks, and `14` ALMA+JWST `[C II]` disks at $z = 4.2\text{--}7.3$).

### 2. Deterministically Generated Paper 21 Outputs (`data/derived/`)
Generated from scratch by `scripts/compute_bh_cosmology_duality_audit.py`:
- `paper21_bh_cosmology_duality_results.json`: Complete machine-readable ledger of all frozen `v20` ($k = 0$) invariants, `SymPy` symbolic proofs (`Theorems D1–D5`), astrophysical and Planckian black hole horizon thermodynamics (`Extremal K4 Seed Remnant`, `10 M_sun`, `GW150914`, `Sgr A*`, `M87*`), non-singular `K4 Seed Genesis` bounce observables, `15` Moresco+/Stern+/Jimenez+ Cosmic Chronometer $H(z)$ points, Euclid DR1 cross-link, and multi-probe observational benchmarks, with `NATURE_LEVEL_CLOSURE = false` strictly enforced.
- `paper21_bh_radial_zones_profile.csv`: Continuous radial profile across $x = r/r_s \in [0.002, 3.50]$ resolving the exterior horizon and all four concentric non-singular interior zones (`Zone I`–`Zone IV`).
- `paper21_cosmic_expansion_history.csv`: Continuous cosmic expansion trajectory across $61$ orders of magnitude from the `K4 Seed Genesis` bounce ($\log_{10} a_{\mathrm{bounce}} = -31.005$) to the present accelerating epoch.
- `paper21_observational_benchmarks.csv`: Tabulated zero-parameter ($k = 0$) validation summary across strong-field black hole shadows, gravitational waves, binary pulsars, Planck PR3 CMB, JWST $z = 14.18$, high-$z$ disk kinematics, Cosmic Chronometers, live ESA Euclid DR1, and the Eridanus Supervoid ISW decrement.
