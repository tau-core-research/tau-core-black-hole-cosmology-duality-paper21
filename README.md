# Paper 21 — Black Holes, the Big Bang, and Present Cosmic Expansion as Dual Painlevé–Gullstrand Regimes of the Discrete $K_4$ Temporal Substrate: Non-Singular Anatomy, Horizon Thermodynamics, and Multi-Probe Validation ($k = 0$)

**Author:** Jolcsak, A. (*Tau Core Research Collaboration*)  
**Repository:** `tau-core-black-hole-cosmology-duality-paper21`  
**Series:** Tau Core Theoretical & Observational Synthesis Series (Paper 21)  
**Knowledge Base Connection:** `.tau-core-kb -> /Volumes/EXTERNAL/Research/Tau Core/tau-core-ai-knowledge-base`  
**Free Tuned Parameters:** $k = 0$ | **Epistemic Gate:** `NATURE_LEVEL_CLOSURE = false`

---

## Main Claim

General Relativity paired with $\Lambda$CDM treats six foundational problems of gravitational physics—(i) central black hole singularities ($r \to 0$) and Cauchy-horizon mass inflation, (ii) the initial Big Bang singularity ($a \to 0$), (iii) the $10^{122}$ quantum-vacuum cosmological-constant catastrophe, (iv) Penrose's past-Weyl horizon-entropy paradox, (v) galactic/cluster "dark matter", and (vi) late-time cosmic "dark energy"—as disconnected anomalies requiring separate auxiliary hypotheses.

In **Paper 21**, we prove analytically (verified symbolically via `SymPy` in `scripts/compute_bh_cosmology_duality_audit.py`) and test across strong-field and cosmological observations that under Tau Core's **Three-Pillar Canonical Synthesis ($k = 0$ free parameters)**:
1. **`Pillar I` (`Theorem U1/F1`):** Algebraic Clifford-projector superselection ($\Pi^2 = \Pi \implies r_{\mathrm{seed}} = 1/4, \dot{r} \equiv 0$), positive $S^2$ Maurer–Cartan stiffness ($\nabla^2_{S^2} V_{\mathrm{MC}} = 4I_2 \succ 0$), and the fundamental simplicial cutoff $\ell_* = \sqrt{24\pi}\,\ell_P \approx 8.6832\,\ell_P = 1.4034 \times 10^{-34}\;\mathrm{m}$ ($M_* = E_P/\sqrt{24\pi} = 1.4060 \times 10^{18}\;\mathrm{GeV}$);
2. **`Pillar II` (`Theorem U2/F2`):** Connection-free Cartan–Kodama exterior 2-form holonomy $\mathrm{d}K$ across both stationary black hole horizons ($\kappa_{\mathrm{BH}} = c^2/(2r_s)$) and non-stationary cosmic apparent horizons ($|\kappa_{\mathrm{KH}}(z)| = \frac{1}{2}cH(z)|1 - q(z)| = cH(z)|1 - \frac{3}{4}\Omega_m(z)|$);
3. **`Pillar III` (`Theorem U3/F3`):** Short-range `COCA-H1` Yukawa core subtraction ($\Phi_{K_4} = \Phi_0 - \Phi_m$), 15-channel `CHDF` UV trace cancellation ($\mathrm{Tr}(T_{\mathrm{def}}^{(15)}) = 0$), and multi-probe observational validation;

black hole interiors and cosmic expansion emerge as **exact sign-reversed regimes ($\sigma_{\mathrm{flow}} = -1$ vs. $\sigma_{\mathrm{flow}} = +1$) of a single Painlevé–Gullstrand (PG) temporal fluid**:

$$\mathrm{d}s^2 = -c^2\,\mathrm{d}t_{\mathrm{PG}}^2 + \bigl(\mathrm{d}r - \sigma_{\mathrm{flow}}\,c\,\beta(t_{\mathrm{PG}}, r)\,\mathrm{d}t_{\mathrm{PG}}\bigr)^2 + r^2\,\mathrm{d}\Omega^2, \qquad \dot{r}_\pm = c\bigl(\pm 1 + \sigma_{\mathrm{flow}}\beta\bigr).$$

### Five Core Theorems (`Theorems D1–D5`) & Multi-Probe Results ($k = 0$)
- **`Theorem D1` (PG Sign-Flip Duality & Cartan Holonomy):** Exact unification of inward black hole PG waterfalls ($\sigma_{\mathrm{flow}} = -1$, $g_{0r}(1/4) = -\sqrt{5}/8 \approx -0.27951$) and outward cosmic PG expansion ($\sigma_{\mathrm{flow}} = +1$) under $\mathrm{d}K$.
- **`Theorem D2` (Cosmic Acceleration = Galactic Dip Identity):** In spatially flat matter + $\Lambda$-asymptote FLRW, cosmic acceleration onset $q(z_{\mathrm{acc}}) = 0$ ($\ddot{a} = 0$) and the Kodama–Hayward galactic acceleration minimum $\mathrm{d}a_0^{\mathrm{KH}}/\mathrm{d}z = 0$ occur at the **exact same cosmic epoch** $\Omega_m(z) = 2/3$:
  $$z_{\mathrm{acc}} \equiv z_{\mathrm{dip}} = \left(\frac{2\Omega_{\Lambda,0}}{\Omega_{m,0}}\right)^{1/3} - 1 = 0.63155650, \qquad \frac{a_0^{\mathrm{KH}}(z_{\mathrm{dip}})}{a_0^{\mathrm{KH}}(0)} = 0.93855026.$$
- **`Theorems D3–D4` (Non-Singular Four-Zone Black Hole Anatomy & $K_4$ Seed Genesis Bounce):** Inside $r < r_s$, collapse traverses **Zone I** ($+2:-1:-1$ traceless `CHDF` tidal stretching, eigenvalues $+\frac{\sqrt{3}}{12}, -\frac{\sqrt{3}}{24}, -\frac{\sqrt{3}}{24}$), **Zone II** (`COCA-H1` cancellation of $1/r^2$ force to finite $+\frac{1}{2}GM m_{\mathrm{COCA}}^2$ & 48-Weyl chiral deconfinement), **Zone III** (`CHDF` $\mathrm{Tr}(T_{\mathrm{def}}^{(15)}) = 0$ Cauchy-horizon brake at $x_- \approx 0.0323$ & $\pi_2(S^2) = \mathbb{Z}$ spin-texture angular-momentum storage), and **Zone IV** ($\Pi^2 = \Pi \implies r_{\mathrm{seed}} = 1/4, \dot{r} = 0$ incompressible de Sitter $K_4$ Seed Core with $f_{\mathrm{eff}}(0) = +1$, $w(0) = -1$, and $\mathcal{K}_{\max}\ell_P^4 = \frac{1}{24\pi^2} \approx 4.22 \times 10^{-3}$). Under $\sigma_{\mathrm{flow}} = +1$, cosmic genesis exhibits a smooth non-singular bounce $H^2(a) = H_{\mathrm{std}}^2(a)[1 - (H_{\mathrm{std}}(a)/H_{\mathrm{crit}})^2]$ at $\log_{10} a_{\mathrm{bounce}} = -31.005$ ($a_{\mathrm{bounce}} = 9.89 \times 10^{-32}$) with peak $H_{\max} = c/(2\ell_*) = 1.07 \times 10^{42}\;\mathrm{s}^{-1}$.
- **`Theorem D5` (Simplicial Plaquette Entropy, Extremal Remnant & Penrose Weyl Resolution):** Local $S_4(\sigma)\times S_4(\sigma')$ gauge locking restricts director modes to horizon cuts, yielding $S_H/k_B = \frac{A_H}{4\ell_P^2} = 6\pi\frac{A_H}{\ell_*^2}$ ($18.85\,k_B$ per $\ell_*^2$ plaquette). The regular Hawking temperature $T_H^{\mathrm{Tau}}(r_+) = \frac{\hbar c}{4\pi k_B r_+}(1 - 3\ell_*^2/r_+^2)$ peaks at $r_+ = 3\ell_*$ ($T_{H,\max} = 2.89 \times 10^{29}\;\mathrm{K}$) and cools with $C_{\mathrm{BH}} > 0$ to $T_H = 0\;\mathrm{K}$ at the extremal $K_4$ Seed Remnant ($r_{\mathrm{ext}} = \sqrt{3}\ell_*$, $M_{\mathrm{rem}} = \frac{9\sqrt{2\pi}}{2}M_P = 11.28\,M_P$, $N_{\mathrm{plaq,rem}} = 12\pi \approx 37.70$, $S_{\mathrm{rem}}/k_B = 72\pi^2 \approx 710.61$), while cosmic horizon entropy grows across $121.08$ orders of magnitude to $S_{\mathrm{CH}}(0)/k_B = 2.27 \times 10^{122}$ from an isotropic $\mathcal{C}_{\mathrm{Weyl}} = 0$ $K_4$ Genesis state.
- **Multi-Probe Validation:** Passes EHT Sgr A* ($52.11\;\mu\mathrm{as}$ vs. $48.7 \pm 7.0\;\mu\mathrm{as}$, $0.49\sigma$), EHT M87* ($39.69\;\mu\mathrm{as}$ vs. $42.0 \pm 3.0\;\mu\mathrm{as}$, $0.77\sigma$; `OCSR` halo onset $\ell_\tau = 3.17\;\mathrm{kpc}$), GW170817 ($c_T = c$), PSR J0737-3039A/B ($3.75 \times 10^{-17}$), GW150914 area theorem, 15 Cosmic Chronometers ($\chi^2/\mathrm{dof} = 0.38$), Planck PR3 `plik_lite_v22` ($\Omega_A/\Omega_b = 5.3681$, $0.06\sigma$ parity), JWST `JADES-GS-z14-0` ($z_{\mathrm{vir}} = 25.8$), 43 high-$z$ disks ($\chi^2 = 138.97$ vs. $\Lambda\mathrm{CDM}$ $227.75$), live ESA Euclid DR1 (`18,983` records, `5,955` unique 3D objects, `359` NISP spectroscopic galaxies with $1.3247\times$ median boost, `2,859` positive-clock void objects peaking at $+484.04\;\mathrm{ppm}$), and Eridanus Supervoid ISW ($-153.79\;\mu\mathrm{K}$ vs. $-146 \pm 18\;\mu\mathrm{K}$, $0.43\sigma$).

---

## Explicit Epistemic Boundaries (`NATURE_LEVEL_CLOSURE = false`)

In strict compliance with `.tau-core-kb/CLAIM_BOUNDARIES.md`, **`NATURE_LEVEL_CLOSURE = false`** is maintained upfront:
1. **`POST-OCC-NU1` / `EAS-SNPR-N1` Ontic Realization & Continuous Neutrino Mass Spectrum:** Unconditional ontic instantiation of $\mathcal{S}_{\mathrm{obs}} \neq \emptyset$ ($|\nu| = 1$ over $\nu = 0$) and first-principles derivation of continuous absolute neutrino mass splittings remain open upstream boundaries.
2. **Non-Perturbative Multi-Vertex $\ell_*$ Bounce Backreaction:** The modified bounce Friedmann equation captures leading simplicial holonomy saturation at $\ell_* = \sqrt{24\pi}\,\ell_P$; full non-perturbative multi-vertex Wetterich PDE integration across the bounce turnaround remains open.
3. **Self-Consistent Quantum Horizon Evaporation Backreaction:** While `Theorem D5` establishes the regular Hayward–Tau temperature curve and elliptic $\Delta_A \Phi_{K_4}$ boundary readout ($D_{OS}^{\mathrm{op}} > 0$), full 4D non-equilibrium greybody + backreaction evolution near $r_+ \sim 3\ell_*$ remains open.
4. **Low-Mass Galaxy Group AGN Gas Evacuation ($M_{\mathrm{halo}} \sim 10^{13}\,M_\odot$, $T_X < 2.5\;\mathrm{keV}$):** Applying the massive-cluster shock containment boundary without potential-depth-dependent AGN gas escape underpredicts low-mass group masses (`HSC-XXL` low-mass subset median $M_{\mathrm{Tau}}/M_{\mathrm{WL}} = 0.679$).

---

## Deterministic Reproduction

Run the complete end-to-end verification, figure generation, bilingual PDF compilation, arXiv packaging, and `pytest` audit from the repository root:

```bash
python3 scripts/reproduce.py
```

### Repository Structure
- `paper21_submission_source/`: English submission manuscript (`main.tex`, `refs.bib`, 7 English 300 DPI `figures/`, compiled `main.pdf`)
- `paper21_hungarian_source/`: Native UTF-8 Hungarian companion manuscript (`main.tex`, `refs.bib`, 7 Hungarian-localized 300 DPI `figures/`, compiled `main.pdf`)
- `arxiv_submission_source.zip`: Clean source-only arXiv package
- `scripts/compute_bh_cosmology_duality_audit.py`: Deterministic `SymPy` + numerical audit engine
- `scripts/generate_paper21_figures.py`: Bilingual 7-figure publication generator
- `tests/test_public_reproducibility_package.py`: Automated verification suite
