"""
run_pipeline.py
===============
F3 Block Multi-Well Rock Physics & Synthetic Seismogram Pipeline
================================================================

Full automated workflow for all wells (all real .wll data).

Phases:
    1  Load & QC      — Parse .wll files (real wells)
    2  Elastic params — AI, Vp/Vs, Poisson's ratio, K_sat, G_sat
    3  Gassmann sub   — Brine → Gas fluid substitution
    4  Seismic        — Zero-offset synth, angle gather, AVO attributes
    5  Multi-well     — Cross-well overlay crossplots and log comparisons

Usage:
    python run_pipeline.py                              # runs all phases, all wells
    python run_pipeline.py --wll_dir ./my_wlls          # custom .wll directory
    python run_pipeline.py --wells F02-1 F03-2          # specific wells only
    python run_pipeline.py --skip_phases 4              # skip seismic phase
    python run_pipeline.py --f_dom 30                   # change wavelet frequency

Wells:
    F02-1  — real .wll files (loaded from --wll_dir)
    F03-2  — real .wll files (loaded from --wll_dir)
    F06-1  — real .wll files (loaded from --wll_dir)

Output structure:
    outputs/
        F02-1/
            01_log_overview.png
            02_elastic_params.png
            03_crossplots.png
            04_fluid_sub.png
            05_zero_offset.png
            06_gather_brine.png
            06_gather_gas.png
            07_diff_avo.png
            08_wavelet_comparison.png
            09_well_tie.png
            rock_physics.csv
        F03-2/   (same structure)
        F06-1/   (same structure)
        multi_well/
            crossplot_brine_gas.png
            log_overlay_VP.png
            log_overlay_GR.png
            log_overlay_AI.png
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from modules.wll_reader      import WellLoader
from modules.rock_physics     import compute_elastic_params, fluid_substitution, elastic_summary
from modules.seismic_engine   import SeismicEngine
from modules.plotting import (
    plot_log_overview, plot_elastic_params, plot_crossplots,
    plot_fluid_substitution, plot_zero_offset, plot_angle_gather,
    plot_difference_avo, plot_wavelet_comparison, plot_well_tie,
    plot_multi_well_crossplot, plot_multi_well_logs,
)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WLL_DIR  = '/mnt/user-data/uploads'
DEFAULT_OUT_DIR  = '/mnt/user-data/outputs'
DEFAULT_WELLS    = ['F02-1', 'F03-2', 'F03-4', 'F06-1']
DEFAULT_F_DOM    = 40.0
DEFAULT_ANGLES   = np.arange(0, 46, 3)
ALL_PHASES       = [1, 2, 3, 4, 5]

# All wells now use real .wll data
REAL_WELLS = ['F02-1', 'F03-2', 'F03-4', 'F06-1']
# Synthetic wells dictionary kept empty — no synthetic generation needed
SYNTHETIC_WELLS = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str, char: str = '═', width: int = 68):
    print(f"\n{char*width}")
    print(f"  {text}")
    print(f"{char*width}")


def phase_header(n: int, title: str):
    print(f"\n  ┌── Phase {n}: {title} {'─'*(55 - len(title))}")


def phase_done(t0: float):
    print(f"  └── done  ({time.time() - t0:.1f}s)")


def well_out_dir(base_out: str, well: str) -> Path:
    p = Path(base_out) / well
    p.mkdir(parents=True, exist_ok=True)
    return p


def p(out_dir: Path, filename: str) -> str:
    return str(out_dir / filename)


# ─────────────────────────────────────────────────────────────────────────────
# PER-WELL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_well(well_name: str, df: pd.DataFrame,
             out_dir: Path, phases: list,
             f_dom: float = DEFAULT_F_DOM,
             angles: np.ndarray = DEFAULT_ANGLES) -> pd.DataFrame:
    """
    Run all enabled phases for a single well.

    Parameters
    ----------
    well_name : str
    df        : input DataFrame (from WLL reader or synthetic generator)
    out_dir   : per-well output directory
    phases    : list of phase numbers to run
    f_dom     : wavelet dominant frequency
    angles    : pre-stack gather angles

    Returns
    -------
    df        : enriched DataFrame with elastic params + Gassmann results
    """
    print(f"\n  {'▶'*3}  Well: {well_name}  "
          f"({'real .wll' if df.attrs.get('source','real') != 'synthetic' else 'synthetic'})"
          f"  —  {len(df)} samples  "
          f"[{df['DEPTH'].min():.0f}–{df['DEPTH'].max():.0f} m]")

    # ── Phase 1: QC overview ──────────────────────────────────────────────
    if 1 in phases:
        t0 = time.time()
        phase_header(1, 'Log QC & Overview')
        plot_log_overview(df, well_name=well_name,
                          save_path=p(out_dir, '01_log_overview.png'))
        phase_done(t0)

    # ── Phase 2: Elastic parameters ───────────────────────────────────────
    if 2 in phases:
        t0 = time.time()
        phase_header(2, 'Elastic Parameters')

        # Ensure VS is present (estimate if missing)
        if 'VS' not in df.columns:
            print('     ⚠  VS not found — estimating from Vp (Vp/Vs = 1.9)')
            df['VS'] = (df['VP'] / 1.9).astype('float32')

        df = compute_elastic_params(df)
        print(f"\n{elastic_summary(df).to_string()}\n")

        plot_elastic_params(df, well_name=well_name,
                            save_path=p(out_dir, '02_elastic_params.png'))
        plot_crossplots(df, well_name=well_name,
                        save_path=p(out_dir, '03_crossplots.png'))
        phase_done(t0)

    # ── Phase 3: Gassmann ─────────────────────────────────────────────────
    if 3 in phases:
        t0 = time.time()
        phase_header(3, 'Gassmann Fluid Substitution (Brine → Gas)')

        phi_col = 'PHIT' if 'PHIT' in df.columns else None
        df = fluid_substitution(df, phi_col=phi_col or 'PHIT',
                                 fluid_in='brine', fluid_out='gas')

        ai_chg   = (df['AI_gas'].mean()   - df['AI'].mean())   / df['AI'].mean()   * 100
        vpvs_chg = (df['VPVS_gas'].mean() - df['VPVS'].mean()) / df['VPVS'].mean() * 100
        print(f"     AI change    : {ai_chg:+.1f}%  "
              f"({df['AI'].mean():.0f} → {df['AI_gas'].mean():.0f} × 10³ kg/m²s)")
        print(f"     Vp/Vs change : {vpvs_chg:+.1f}%  "
              f"({df['VPVS'].mean():.3f} → {df['VPVS_gas'].mean():.3f})")

        plot_fluid_substitution(df, well_name=well_name,
                                save_path=p(out_dir, '04_fluid_sub.png'))
        phase_done(t0)

    # ── Phase 4: Seismic ──────────────────────────────────────────────────
    if 4 in phases:
        t0 = time.time()
        phase_header(4, 'Synthetic Seismograms')

        eng = SeismicEngine(df, f_dom=f_dom, wavelet_type='ricker', angles=angles)
        results = eng.run()

        # 4a: Zero-offset
        plot_zero_offset(df,
                         results['zero_brine'], results['zero_gas'],
                         well_name=well_name,
                         save_path=p(out_dir, '05_zero_offset.png'))

        # 4b: Angle gathers
        plot_angle_gather(results['gather_brine'], well_name=well_name,
                          case='brine',
                          save_path=p(out_dir, '06_gather_brine.png'))
        plot_angle_gather(results['gather_gas'], well_name=well_name,
                          case='gas',
                          save_path=p(out_dir, '06_gather_gas.png'))

        # 4c: Difference + AVO attributes
        plot_difference_avo(results['gather_brine'], results['gather_gas'],
                             results['avo_attrs'],
                             well_name=well_name,
                             save_path=p(out_dir, '07_diff_avo.png'))

        # 4d: Wavelet comparison
        plot_wavelet_comparison(df, SeismicEngine,
                                 well_name=well_name,
                                 save_path=p(out_dir, '08_wavelet_comparison.png'))

        # 4e: Well tie
        plot_well_tie(df, results['zero_brine'], results['zero_gas'],
                       well_name=well_name,
                       save_path=p(out_dir, '09_well_tie.png'))

        phase_done(t0)

    # Save CSV
    csv_path = p(out_dir, 'rock_physics.csv')
    cols = [c for c in df.columns if c not in ('LITH', 'LITH_DETAIL')]
    df[cols].round(5).to_csv(csv_path, index=False)
    print(f"     📄 CSV → {csv_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-WELL PHASE
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_well(all_wells: dict, out_base: str, phases: list):
    """Phase 5: cross-well overlay figures."""
    if 5 not in phases:
        return

    t0 = time.time()
    mw_dir = Path(out_base) / 'multi_well'
    mw_dir.mkdir(parents=True, exist_ok=True)

    phase_header(5, 'Multi-Well Overlays')

    # Check which wells have the necessary columns
    eligible = {w: df for w, df in all_wells.items()
                if 'AI' in df.columns and 'VPVS' in df.columns}

    if len(eligible) < 2:
        print('     ⚠  Need ≥ 2 wells with elastic params for multi-well plots')
        return

    plot_multi_well_crossplot(eligible,
                               save_path=str(mw_dir / 'crossplot_brine_gas.png'))

    for log in ['VP', 'GR', 'AI', 'VPVS']:
        plot_multi_well_logs(eligible, log_col=log,
                              save_path=str(mw_dir / f'log_overlay_{log}.png'))
        print(f"  → {log} overlay")

    phase_done(t0)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='F3 Block Multi-Well Rock Physics & Seismic Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('--wll_dir', default=DEFAULT_WLL_DIR,
                        help='Directory containing .wll files  '
                             f'(default: {DEFAULT_WLL_DIR})')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                        help=f'Output directory  (default: {DEFAULT_OUT_DIR})')
    parser.add_argument('--wells', nargs='+', default=DEFAULT_WELLS,
                        help='Wells to process  (default: F02-1 F03-2 F06-1)')
    parser.add_argument('--skip_phases', nargs='+', type=int, default=[],
                        help='Phase numbers to skip (1–5)')
    parser.add_argument('--only_phases', nargs='+', type=int, default=[],
                        help='Run only these phases (overrides skip_phases)')
    parser.add_argument('--f_dom', type=float, default=DEFAULT_F_DOM,
                        help=f'Wavelet dominant frequency Hz (default: {DEFAULT_F_DOM})')
    parser.add_argument('--no_real', action='store_true',
                        help='Skip real .wll files (synthetic wells only)')
    parser.add_argument('--no_synthetic', action='store_true',
                        help='Skip synthetic well generation (real wells only)')

    args = parser.parse_args()

    # Determine phases to run
    if args.only_phases:
        phases = sorted(args.only_phases)
    else:
        phases = [p for p in ALL_PHASES if p not in args.skip_phases]

    banner(f"F3 Block Multi-Well Rock Physics & Seismic Pipeline")
    print(f"  Wells    : {', '.join(args.wells)}")
    print(f"  Phases   : {phases}")
    print(f"  WLL dir  : {args.wll_dir}")
    print(f"  Output   : {args.out_dir}")
    print(f"  Wavelet  : Ricker {args.f_dom} Hz")

    # ── Load real wells ───────────────────────────────────────────────────
    all_wells = {}

    real_requested = [w for w in args.wells if w in REAL_WELLS]
    if real_requested and not args.no_real:
        print(f"\n  Loading real .wll files from: {args.wll_dir}")
        loader = WellLoader(wll_dir=args.wll_dir, verbose=True)
        loader.discover_wells()
        loader.print_inventory()

        for wname in real_requested:
            try:
                df = loader.load_well(wname, qc=True)
                df.attrs['source'] = 'real'
                all_wells[wname] = df
            except KeyError as e:
                print(f"  ⚠  {e}  (skipping)")

    if not all_wells:
        print("\n  ✗  No wells loaded. Exiting.")
        sys.exit(1)

    banner(f"Processing {len(all_wells)} well(s)")

    # ── Per-well pipeline ─────────────────────────────────────────────────
    processed = {}
    for wname, df in all_wells.items():
        out_dir = well_out_dir(args.out_dir, wname)
        enriched = run_well(wname, df, out_dir, phases,
                             f_dom=args.f_dom,
                             angles=DEFAULT_ANGLES)
        processed[wname] = enriched

    # ── Multi-well phase ──────────────────────────────────────────────────
    run_multi_well(processed, args.out_dir, phases)

    # ── Final summary ─────────────────────────────────────────────────────
    banner("Pipeline Complete ✅")
    print(f"\n  Wells processed  : {', '.join(processed.keys())}")
    print(f"  Outputs written  : {args.out_dir}")
    print()


if __name__ == '__main__':
    main()
