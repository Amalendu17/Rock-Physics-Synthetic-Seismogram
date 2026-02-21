"""
modules/synthetic_wells.py
==========================
Synthetic Well Log Generator for F3 Block Wells Without .wll Files
F3 Block Multi-Well Rock Physics Framework

Generates geologically realistic synthetic logs for F03-2 and F06-1
based on published F3 Block North Sea subsurface characterisation.

Key references used for calibration:
  - Gausland (2000): F3 Block stratigraphy and well correlation
  - Sørensen et al. (1997): North Sea Late Cretaceous basin analysis
  - OpendTect F3 Demo dataset documentation

F3 Block Geology summary:
  The F3 Block contains well-documented Cenozoic clastic sequences.
  Three wells are commonly used:  F02-1, F03-2, F06-1.
  Reservoir: Pliocene-Pleistocene unconsolidated sands (gas-bearing).
  Seal: Holocene marine clays.
  The sand body shows classic Class III AVO anomaly.

Well differences:
  F02-1: Western flank. Deeper water depth. Lower GR sands (~750-900m).
  F03-2: Central. Slightly shallower gas sand (~700-850m). Good VP/VS coverage.
  F06-1: Eastern flank. Thinner gas sand. Higher clay content (higher GR).

Usage:
    from modules.synthetic_wells import generate_f03, generate_f06, generate_all_synthetic
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(12345)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, k: int = 15) -> np.ndarray:
    """Simple running-mean smoother."""
    kernel = np.ones(k) / k
    return np.convolve(arr, kernel, mode='same')


def _build_log(depth, zones, noise_std, seed_offset=0):
    """
    Build a log value array from depth zones with realistic noise and
    gradual transitions (5 m ramp) between lithologies.

    zones : list of (d_top, d_bot, value)
    """
    rng_local = np.random.default_rng(42 + seed_offset)
    out = np.full(len(depth), np.nan)
    for d_top, d_bot, val in zones:
        m = (depth >= d_top) & (depth < d_bot)
        out[m] = val

    # Forward-fill any gaps (shale default)
    for i in range(len(out)):
        if np.isnan(out[i]) and i > 0:
            out[i] = out[i-1]
    out = np.nan_to_num(out, nan=np.nanmean(out))

    # Gaussian noise
    out += rng_local.normal(0, noise_std, len(out))

    # Smooth transitions
    out = _smooth(out, k=11)
    return out


def _compaction(depth, start, rate=0.10):
    """Linear compaction gradient."""
    return rate * (depth - start) / (depth[-1] - start)


# ─────────────────────────────────────────────────────────────────────────────
# F03-2  (Central well — best gas sand development)
# ─────────────────────────────────────────────────────────────────────────────

def generate_f03(n_samples: int = 9500,
                 depth_start: float = 50.0,
                 depth_step:  float = 0.15) -> pd.DataFrame:
    """
    Generate synthetic logs for F03-2 (central F3 Block well).

    Geology:
        50  – 650 m : Upper Cenozoic marine clays (shale)
        650 – 720 m : Transition / silty sands
        720 – 870 m : GAS SAND reservoir (main target)
        870 – 950 m : Brine sand transition
        950 – 1450m : Lower Cenozoic clays (compacted shale)

    Calibrated to F3 Block published ranges:
        Shale  Vp ~ 1900–2400 m/s,  Vs ~ 800–1100 m/s, ρ ~ 2.1–2.3 g/cc
        Brine sand Vp ~ 2300–2600,   Vs ~ 1400–1600,    ρ ~ 2.0–2.2 g/cc
        Gas sand   Vp ~ 1600–2100,   Vs ~ 1300–1500,    ρ ~ 1.85–2.05 g/cc
    """
    depth = np.arange(n_samples) * depth_step + depth_start
    comp  = _compaction(depth, depth_start, rate=0.12)

    # ── Lithology zones ───────────────────────────────────────────────────
    lith = np.zeros(len(depth), dtype=np.int8)           # 0 = shale
    lith[(depth >= 720) & (depth < 870)] = 2             # gas sand
    lith[(depth >= 870) & (depth < 950)] = 1             # brine sand
    lith[(depth >= 650) & (depth < 720)] = 1             # silt/sand transition

    # ── Vp (m/s) ─────────────────────────────────────────────────────────
    vp_zones = [
        (50,   650,  1950.0),   # shallow shale
        (650,  720,  2200.0),   # silt transition
        (720,  870,  1850.0),   # GAS SAND — low Vp
        (870,  950,  2380.0),   # brine sand
        (950,  1500, 2450.0),   # deep shale
    ]
    vp = _build_log(depth, vp_zones, noise_std=45, seed_offset=1)
    vp += comp * 180

    # ── Vs (m/s) ─────────────────────────────────────────────────────────
    vs_zones = [
        (50,   650,   870.0),
        (650,  720,  1250.0),
        (720,  870,  1380.0),   # gas sand — Vs relatively unchanged
        (870,  950,  1520.0),
        (950,  1500, 1100.0),
    ]
    vs = _build_log(depth, vs_zones, noise_std=35, seed_offset=2)
    vs += comp * 70

    # ── RHOB (g/cc) ──────────────────────────────────────────────────────
    rho_zones = [
        (50,   650,  2.18),
        (650,  720,  2.12),
        (720,  870,  1.96),   # gas sand — lower density
        (870,  950,  2.14),
        (950,  1500, 2.24),
    ]
    rhob = _build_log(depth, rho_zones, noise_std=0.018, seed_offset=3)
    rhob += comp * 0.035

    # ── GR (API) ─────────────────────────────────────────────────────────
    gr_zones = [
        (50,   650,  95.0),
        (650,  720,  48.0),
        (720,  870,  25.0),
        (870,  950,  32.0),
        (950,  1500, 88.0),
    ]
    gr = _build_log(depth, gr_zones, noise_std=5.5, seed_offset=4)
    gr = gr.clip(1, 200)

    # ── PHIT (fraction) ──────────────────────────────────────────────────
    phi_zones = [
        (50,   650,  0.09),
        (650,  720,  0.22),
        (720,  870,  0.31),   # gas sand — high porosity
        (870,  950,  0.26),
        (950,  1500, 0.07),
    ]
    phit = _build_log(depth, phi_zones, noise_std=0.012, seed_offset=5)
    phit = phit.clip(0.02, 0.50)

    # ── DT (us/ft) ───────────────────────────────────────────────────────
    # DT ≈ 304800 / Vp_ft   (1 ft = 0.3048 m, vp in ft/s → us/ft)
    dt = 1e6 / (vp / 0.3048)   # us/ft

    # ── LITHOLOGY mapping ────────────────────────────────────────────────
    lithology = lith.copy().astype(int)  # 0=shale, 1=brine, 2=gas (for compatibility)
    # Collapse to binary for pipeline (0=shale, 1=sand)
    lith_binary = np.where(lithology > 0, 1, 0).astype(np.int8)

    df = pd.DataFrame({
        'DEPTH'     : depth.astype(np.float32),
        'VP'        : vp.astype(np.float32),
        'VS'        : vs.astype(np.float32),
        'RHOB'      : rhob.astype(np.float32),
        'GR'        : gr.astype(np.float32),
        'PHIT'      : phit.astype(np.float32),
        'DT'        : dt.astype(np.float32),
        'LITHOLOGY' : lith_binary,
        'LITH_DETAIL': lithology,   # 0=shale, 1=brine, 2=gas
    })

    df.attrs['well']   = 'F03-2'
    df.attrs['source'] = 'synthetic'
    return df


# ─────────────────────────────────────────────────────────────────────────────
# F06-1  (Eastern flank — thinner, more clay-rich sand)
# ─────────────────────────────────────────────────────────────────────────────

def generate_f06(n_samples: int = 8000,
                 depth_start: float = 50.0,
                 depth_step:  float = 0.15) -> pd.DataFrame:
    """
    Generate synthetic logs for F06-1 (eastern flank well).

    Geology:
        50  – 600 m : Upper Cenozoic marine clays
        600 – 680 m : Sandy transition (higher clay content than F03)
        680 – 780 m : GAS SAND (thinner than F03; more clay → higher GR)
        780 – 860 m : Brine sand / water-wet transition
        860 – 1250m : Compacted Cenozoic clays

    Eastern flank characteristics:
        - Shallower target (~730 m vs 795 m in F02-1)
        - Thinner gas column (~100 m vs ~150 m)
        - Slightly higher clay content → higher GR, lower Vp/Vs contrast
        - Gas sand Vp slightly higher than F03 (lower porosity)
    """
    depth = np.arange(n_samples) * depth_step + depth_start
    comp  = _compaction(depth, depth_start, rate=0.09)

    # ── Lithology ─────────────────────────────────────────────────────────
    lith = np.zeros(len(depth), dtype=np.int8)
    lith[(depth >= 680) & (depth < 780)] = 2    # gas sand
    lith[(depth >= 780) & (depth < 860)] = 1    # brine sand
    lith[(depth >= 600) & (depth < 680)] = 1    # sandy transition

    # ── Vp ───────────────────────────────────────────────────────────────
    vp_zones = [
        (50,   600,  1880.0),
        (600,  680,  2150.0),
        (680,  780,  2000.0),   # gas sand — slightly higher Vp than F03
        (780,  860,  2350.0),
        (860,  1250, 2420.0),
    ]
    vp = _build_log(depth, vp_zones, noise_std=50, seed_offset=11)
    vp += comp * 160

    # ── Vs ───────────────────────────────────────────────────────────────
    vs_zones = [
        (50,   600,   840.0),
        (600,  680,  1180.0),
        (680,  780,  1320.0),
        (780,  860,  1490.0),
        (860,  1250, 1080.0),
    ]
    vs = _build_log(depth, vs_zones, noise_std=30, seed_offset=12)
    vs += comp * 60

    # ── RHOB ─────────────────────────────────────────────────────────────
    rho_zones = [
        (50,   600,  2.17),
        (600,  680,  2.10),
        (680,  780,  2.02),    # gas sand — less porous than F03
        (780,  860,  2.15),
        (860,  1250, 2.22),
    ]
    rhob = _build_log(depth, rho_zones, noise_std=0.016, seed_offset=13)
    rhob += comp * 0.030

    # ── GR (higher clay → higher GR in sands) ────────────────────────────
    gr_zones = [
        (50,   600, 100.0),
        (600,  680,  60.0),    # clayey sand
        (680,  780,  38.0),    # gas sand (cleaner but still some clay)
        (780,  860,  42.0),
        (860,  1250, 92.0),
    ]
    gr = _build_log(depth, gr_zones, noise_std=6.0, seed_offset=14)
    gr = gr.clip(1, 200)

    # ── PHIT ─────────────────────────────────────────────────────────────
    phi_zones = [
        (50,   600,  0.08),
        (600,  680,  0.19),
        (680,  780,  0.27),    # slightly lower porosity than F03
        (780,  860,  0.23),
        (860,  1250, 0.07),
    ]
    phit = _build_log(depth, phi_zones, noise_std=0.011, seed_offset=15)
    phit = phit.clip(0.02, 0.50)

    dt = 1e6 / (vp / 0.3048)

    lith_binary = np.where(lith > 0, 1, 0).astype(np.int8)

    df = pd.DataFrame({
        'DEPTH'     : depth.astype(np.float32),
        'VP'        : vp.astype(np.float32),
        'VS'        : vs.astype(np.float32),
        'RHOB'      : rhob.astype(np.float32),
        'GR'        : gr.astype(np.float32),
        'PHIT'      : phit.astype(np.float32),
        'DT'        : dt.astype(np.float32),
        'LITHOLOGY' : lith_binary,
        'LITH_DETAIL': lith.astype(int),
    })

    df.attrs['well']   = 'F06-1'
    df.attrs['source'] = 'synthetic'
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_synthetic() -> dict:
    """
    Returns dict of all synthetic wells:
        { 'F03-2': DataFrame, 'F06-1': DataFrame }
    """
    return {
        'F03-2': generate_f03(),
        'F06-1': generate_f06(),
    }
