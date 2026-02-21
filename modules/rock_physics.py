"""
modules/rock_physics.py
=======================
Elastic Parameters & Gassmann Fluid Substitution
F3 Block Multi-Well Rock Physics Framework

Phase 2:  Baseline elastic parameter calculation
            AI, SI, Vp/Vs, Poisson's ratio, K_sat, G_sat, M_sat

Phase 3:  Gassmann (1951) fluid substitution
            Brine → Gas / Oil / Dry
            Full 3-step workflow: back-strip → forward substitute → recompute

References:
    Gassmann (1951) Uber die elastizitat poroser medien
    Smith, Sondergeld, Rai (2003) GEOPHYSICS
    Mavko, Mukerji, Dvorkin (2009) The Rock Physics Handbook

Usage:
    from modules.rock_physics import compute_elastic_params, fluid_substitution
    df = compute_elastic_params(df)
    df = fluid_substitution(df, fluid_in='brine', fluid_out='gas')
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# FLUID AND MINERAL DATABASES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FluidProps:
    name   : str
    K_fl   : float   # Bulk modulus GPa
    rho_fl : float   # Density g/cc

@dataclass
class MineralProps:
    name   : str
    K_min  : float   # Bulk modulus GPa
    G_min  : float   # Shear modulus GPa
    rho_min: float   # Density g/cc


FLUIDS = {
    'brine': FluidProps('Brine (formation water)', 2.80, 1.08),
    'oil'  : FluidProps('Oil (light crude)',        0.85, 0.80),
    'gas'  : FluidProps('Dry gas (~150 bar)',        0.04, 0.20),
    'dry'  : FluidProps('Dry / air',                0.00, 0.001),
}

MINERALS = {
    'quartz'  : MineralProps('Quartz',    36.6, 44.0, 2.65),
    'calcite' : MineralProps('Calcite',   76.8, 32.0, 2.71),
    'dolomite': MineralProps('Dolomite',  94.9, 45.0, 2.87),
    'clay'    : MineralProps('Clay',       6.9,  2.2, 2.58),
    'feldspar': MineralProps('Feldspar',  37.5, 15.0, 2.63),
}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — ELASTIC PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_elastic_params(df: pd.DataFrame,
                            vp_col  : str = 'VP',
                            vs_col  : str = 'VS',
                            rho_col : str = 'RHOB') -> pd.DataFrame:
    """
    Compute standard rock physics elastic parameters from Vp, Vs, density.

    Adds columns:
        AI     : Acoustic impedance   [10³ kg/m² s]  = ρ·Vp
        SI     : Shear  impedance     [10³ kg/m² s]  = ρ·Vs
        VPVS   : Vp/Vs ratio          [dimensionless]
        PR     : Poisson's ratio      [dimensionless]
        K_SAT  : Saturated bulk mod   [GPa]
        G_SAT  : Shear modulus        [GPa]  (= G_dry)
        M_SAT  : P-wave modulus       [GPa]  = K + 4G/3

    Unit notes:
        RHOB in g/cc × 1000 → kg/m³
        Vp, Vs in m/s
        Moduli in Pa / 1e9 → GPa
    """
    df   = df.copy()
    vp   = df[vp_col].values.astype(float)
    vs   = df[vs_col].values.astype(float)
    rho  = df[rho_col].values.astype(float) * 1000.0   # kg/m³

    # Guard against zero VS (avoids division errors)
    vs = np.where(vs < 200.0, vp / 2.0, vs)

    df['AI']    = (rho * vp) / 1e3
    df['SI']    = (rho * vs) / 1e3
    df['VPVS']  = vp / vs
    r2          = (vp / vs) ** 2
    df['PR']    = (r2 - 2.0) / (2.0 * r2 - 2.0)
    df['G_SAT'] = (rho * vs**2) / 1e9
    df['M_SAT'] = (rho * vp**2) / 1e9
    df['K_SAT'] = df['M_SAT'] - (4.0 / 3.0) * df['G_SAT']

    return df


def elastic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean elastic params grouped by LITHOLOGY (0=shale, 1=sand)."""
    lith_names = {0: 'Shale', 1: 'Sand'}
    cols = [c for c in ['VP','VS','VPVS','PR','AI','SI','K_SAT','G_SAT']
            if c in df.columns]
    summary = df.groupby('LITHOLOGY')[cols].mean().round(3)
    summary.index = summary.index.map(lambda x: lith_names.get(x, str(x)))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — GASSMANN FLUID SUBSTITUTION
# ─────────────────────────────────────────────────────────────────────────────

def _backstrip(K_sat, G_sat, K_fl_orig, K_min, phi) -> np.ndarray:
    """
    Invert Gassmann to recover dry-frame bulk modulus.

        K_dry = (K_sat (φ K_min/K_fl + 1 - φ) - K_min)
                ─────────────────────────────────────────
                 φ K_min/K_fl + K_sat/K_min - 1 - φ
    """
    A     = phi * K_min / (K_fl_orig + 1e-12)
    num   = K_sat * (A + 1.0 - phi) - K_min
    den   = A + K_sat / (K_min + 1e-12) - 1.0 - phi
    K_dry = num / (den + 1e-12)
    return K_dry.clip(0.0, K_min)


def _forward_gassmann(K_dry, G_dry, K_min, K_fl_new, phi) -> np.ndarray:
    """
    Forward Gassmann:
        K_sat = K_dry + (1 - K_dry/K_min)²
                ─────────────────────────────────
                 φ/K_fl + (1-φ)/K_min - K_dry/K_min²
    """
    dK    = 1.0 - K_dry / (K_min + 1e-12)
    den   = phi / (K_fl_new + 1e-12) + (1.0 - phi) / K_min - K_dry / K_min**2
    return (K_dry + dK**2 / (den + 1e-12)).clip(0.0, K_min * 2.5)


def fluid_substitution(df         : pd.DataFrame,
                        phi_col    : str = 'PHIT',
                        fluid_in   : str = 'brine',
                        fluid_out  : str = 'gas',
                        mineral    : str = 'quartz',
                        tag        : str = None) -> pd.DataFrame:
    """
    Full Gassmann fluid substitution on a rock physics DataFrame.

    Requires columns: VP, VS, RHOB, K_SAT, G_SAT, and a porosity column.

    Steps:
      1. Back-strip original fluid  → K_dry
      2. Forward substitute new fluid → K_sat_new
      3. Update density (mass balance)
      4. Recompute Vp, Vs, AI, Vp/Vs, Poisson's ratio

    Parameters
    ----------
    df        : DataFrame with elastic params already computed
    phi_col   : porosity column name (fraction)
    fluid_in  : key in FLUIDS dict for original fluid
    fluid_out : key in FLUIDS dict for new fluid
    mineral   : key in MINERALS dict
    tag       : suffix for output columns (default = fluid_out key)

    Returns
    -------
    DataFrame with new columns:  VP_<tag>, VS_<tag>, RHOB_<tag>,
                                  AI_<tag>, VPVS_<tag>, PR_<tag>,
                                  K_dry (dry-frame modulus)
    """
    df    = df.copy()
    fl_i  = FLUIDS[fluid_in]
    fl_o  = FLUIDS[fluid_out]
    mn    = MINERALS[mineral]
    tag   = tag or fluid_out

    # ── Porosity ──────────────────────────────────────────────────────────
    if phi_col in df.columns:
        phi = df[phi_col].values.astype(float).clip(0.02, 0.45)
    else:
        # Density-derived porosity (mass balance)
        phi = ((mn.rho_min - df['RHOB'].values) /
               (mn.rho_min - fl_i.rho_fl + 1e-6)).clip(0.02, 0.45)
        df['PHIT'] = phi.astype(np.float32)

    K_sat   = df['K_SAT'].values.astype(float)
    G_sat   = df['G_SAT'].values.astype(float)
    rho_old = df['RHOB'].values.astype(float)

    # 1. Back-strip
    K_dry = _backstrip(K_sat, G_sat, fl_i.K_fl, mn.K_min, phi)

    # 2. Forward Gassmann
    K_sat_new = _forward_gassmann(K_dry, G_sat, mn.K_min, fl_o.K_fl, phi)

    # 3. Density
    rho_dry = rho_old - phi * fl_i.rho_fl
    rho_new = (rho_dry + phi * fl_o.rho_fl).clip(1.4, 3.0)

    # 4. Velocities & derived
    rho_si  = rho_new * 1000.0                              # kg/m³
    M_new   = K_sat_new + (4.0 / 3.0) * G_sat              # P-wave modulus GPa
    vp_new  = np.sqrt(np.maximum(M_new  * 1e9 / rho_si, 1.0))
    vs_new  = np.sqrt(np.maximum(G_sat  * 1e9 / rho_si, 1.0))

    ai_new   = (rho_si * vp_new) / 1e3
    vpvs_new = vp_new / vs_new
    r2       = vpvs_new ** 2
    pr_new   = (r2 - 2.0) / (2.0 * r2 - 2.0)

    df[f'VP_{tag}']   = vp_new.astype(np.float32)
    df[f'VS_{tag}']   = vs_new.astype(np.float32)
    df[f'RHOB_{tag}'] = rho_new.astype(np.float32)
    df[f'AI_{tag}']   = ai_new.astype(np.float32)
    df[f'VPVS_{tag}'] = vpvs_new.astype(np.float32)
    df[f'PR_{tag}']   = pr_new.astype(np.float32)
    df['K_dry']       = K_dry.astype(np.float32)

    return df


def substitution_summary(df: pd.DataFrame, tag: str = 'gas') -> pd.DataFrame:
    """
    Print % change in key elastic properties after fluid substitution.
    Returns summary DataFrame.
    """
    rows = []
    for col, unit in [('VP','m/s'), ('AI','10³ kg/m²s'),
                      ('VPVS',''), ('PR',''), ('RHOB','g/cc')]:
        c_orig = col
        c_new  = f'{col}_{tag}'
        if c_orig not in df.columns or c_new not in df.columns:
            continue
        orig  = df[c_orig].mean()
        new   = df[c_new].mean()
        delta = (new - orig) / (abs(orig) + 1e-12) * 100
        rows.append({'Parameter': col, 'Unit': unit,
                     'Original': round(orig, 3),
                     'Substituted': round(new, 3),
                     'Change_%': round(delta, 2)})
    return pd.DataFrame(rows).set_index('Parameter')
