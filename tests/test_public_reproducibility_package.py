from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent
DERIVED_JSON = REPO_ROOT / "data" / "derived" / "paper21_bh_cosmology_duality_results.json"

FIGURE_NAMES = [
    "fig1_pg_duality_and_lightcone_architecture.png",
    "fig2_black_hole_four_zone_interior_anatomy.png",
    "fig3_big_bang_bounce_to_present_expansion.png",
    "fig4_horizon_thermodynamics_entropy_and_page_curve.png",
    "fig5_eht_ligo_and_strong_field_bh_validation.png",
    "fig6_cosmic_expansion_cmb_jwst_euclid_validation.png",
    "fig7_master_bh_vs_cosmos_synthesis_matrix.png",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_files_and_kb_link_exist() -> None:
    required = [
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "DATA_NOTICE.md",
        REPO_ROOT / "LICENSE",
        REPO_ROOT / "README.md",
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "scripts" / "compute_bh_cosmology_duality_audit.py",
        REPO_ROOT / "scripts" / "generate_paper21_figures.py",
        REPO_ROOT / "scripts" / "build_arxiv_source.py",
        REPO_ROOT / "scripts" / "reproduce.py",
        REPO_ROOT / "data" / "raw" / "euclid_dr1_live_validation_results.json",
        REPO_ROOT / "data" / "raw" / "frontier_three_pillars_results.json",
        DERIVED_JSON,
        REPO_ROOT / "data" / "derived" / "paper21_bh_radial_zones_profile.csv",
        REPO_ROOT / "data" / "derived" / "paper21_cosmic_expansion_history.csv",
        REPO_ROOT / "data" / "derived" / "paper21_observational_benchmarks.csv",
    ]
    for path in required:
        assert path.exists(), f"Missing required file: {path}"


def test_sympy_duality_and_zero_parameter_invariants() -> None:
    data = json.loads(DERIVED_JSON.read_text(encoding="utf-8"))
    assert data["free_parameters_k"] == 0
    assert data["NATURE_LEVEL_CLOSURE"] is False

    inv = data["frozen_invariants"]
    assert math.isclose(inv["r_seed"], 0.25, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(inv["alpha_OCSR"], 5.0 * math.sqrt(3.0) / 24.0, rel_tol=1e-12)
    assert math.isclose(inv["omega_A_over_b"], (213.0 * math.sqrt(3.0) - 20.0) / 65.0, rel_tol=1e-12)
    assert math.isclose(inv["plaquette_entropy_kB"], 6.0 * math.pi, rel_tol=1e-12)

    thm = data["sympy_duality_and_coincidence_theorems"]
    assert thm["bh_exterior_holonomy_kappa_c2_over_2rs_exact"] is True
    assert thm["cosmic_exterior_holonomy_dK_exact"] is True
    assert thm["acceleration_dip_coincidence_q0_equals_da0dz0_exact"] is True
    assert thm["chdf_spatial_tidal_2_minus1_minus1_traceless_exact"] is True
    assert thm["chdf_15_channel_uv_trace_zero_exact"] is True
    assert thm["coca_yukawa_cancels_1_over_r2_force_exact"] is True
    assert thm["hayward_hawking_temp_identity_exact"] is True
    assert math.isclose(thm["z_acc_equals_z_dip"], 0.6315564992697738, rel_tol=1e-9)
    assert math.isclose(thm["pg_seed_lightcone_tilt_g0r"], -math.sqrt(5.0) / 8.0, rel_tol=1e-12)

    rem = data["black_hole_interior_and_strong_field"]["k4_seed_remnant"]
    assert rem["T_hawking_extremal_K"] == 0.0
    assert math.isclose(rem["n_plaquettes"], 12.0 * math.pi, rel_tol=1e-12)
    assert math.isclose(rem["S_remnant_over_kB"], 72.0 * math.pi**2, rel_tol=1e-12)


def test_bilingual_figures_exist_and_are_localized() -> None:
    for fname in FIGURE_NAMES:
        main_fig = REPO_ROOT / "figures" / fname
        en_fig = REPO_ROOT / "paper21_submission_source" / "figures" / fname
        hu_fig = REPO_ROOT / "paper21_hungarian_source" / "figures" / fname
        assert main_fig.exists() and main_fig.stat().st_size > 100_000
        assert en_fig.exists() and en_fig.stat().st_size > 100_000
        assert hu_fig.exists() and hu_fig.stat().st_size > 100_000
        assert _sha256(main_fig) == _sha256(en_fig)
        assert _sha256(en_fig) != _sha256(hu_fig), f"Hungarian figure {fname} must be genuinely localized!"


def test_bilingual_manuscripts_and_latex_logs_clean() -> None:
    for folder_name in ("paper21_submission_source", "paper21_hungarian_source"):
        folder = REPO_ROOT / folder_name
        tex_path = folder / "main.tex"
        pdf_path = folder / "main.pdf"
        log_path = folder / "main.log"
        assert tex_path.exists()
        tex_text = tex_path.read_text(encoding="utf-8")
        assert "NATURE" in tex_text and "false" in tex_text
        assert "-1/(2\\sqrt{5})" not in tex_text

        assert pdf_path.exists() and pdf_path.stat().st_size > 500_000, f"Missing or tiny PDF: {pdf_path}"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            assert "LaTeX Warning: Reference" not in log_text
            assert "LaTeX Warning: Citation" not in log_text
            assert "There were undefined references" not in log_text
            overfulls = [float(m) for m in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)]
            severe = [pt for pt in overfulls if pt > 5.0]
            assert not severe, f"Severe Overfull \\hbox in {folder_name}: {severe}"


def test_arxiv_submission_zip_integrity() -> None:
    zip_path = REPO_ROOT / "arxiv_submission_source.zip"
    assert zip_path.exists() and zip_path.stat().st_size > 500_000
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "main.tex" in names
        assert "refs.bib" in names
        for fname in FIGURE_NAMES:
            assert f"figures/{fname}" in names
        assert "main.pdf" not in names
