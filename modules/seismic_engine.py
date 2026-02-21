"""
modules/seismic_engine.py
=========================
Synthetic Seismogram Engine
F3 Block Multi-Well Rock Physics Framework

Implements:
    Wavelet factory          : Ricker, Ormsby bandpass
    Depth → TWT conversion   : interval velocity integration
    Reflection coefficients  : zero-offset and Shuey 2-term AVO
    Synthetic convolution    : zero-offset and pre-stack angle gathers
    AVO attribute extraction : Intercept (R0) and Gradient (G) per interface

All outputs are in TWT (two-way travel time, seconds) domain.

Usage:
    from modules.seismic_engine import SeismicEngine
    eng = SeismicEngine(df, wavelet_type='ricker', f_dom=40)
    result = eng.run()
"""

import numpy as np
import pandas as pd
from scipy.signal import convolve


# ─────────────────────────────────────────────────────────────────────────────
# WAVELET FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def ricker(f_dom: float, dt: float, length_s: float = 0.12) -> np.ndarray:
    """
    Zero-phase Ricker (Mexican-hat) wavelet.

    Parameters
    ----------
    f_dom    : dominant frequency Hz
    dt       : sample interval s
    length_s : wavelet duration s

    Returns
    -------
    w : normalised wavelet array
    """
    n = int(length_s / dt) | 1   # force odd
    t = np.linspace(-(n // 2) * dt, (n // 2) * dt, n)
    pft = (np.pi * f_dom * t) ** 2
    w   = (1.0 - 2.0 * pft) * np.exp(-pft)
    return w / (np.max(np.abs(w)) + 1e-12)


def ormsby(f1: float, f2: float, f3: float, f4: float,
           dt: float, length_s: float = 0.15) -> np.ndarray:
    """
    Zero-phase Ormsby bandpass wavelet  [f1-f2-f3-f4] Hz.
    Uses the analytic trapezoid amplitude-spectrum approach.

    Parameters
    ----------
    f1, f2 : low-cut taper corners Hz
    f3, f4 : high-cut taper corners Hz
    dt     : sample interval s
    """
    n = int(length_s / dt) | 1
    t = np.linspace(-(n // 2) * dt, (n // 2) * dt, n)

    def _sinc2(f, t):
        x = np.pi * f * t
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(x) < 1e-10, 1.0, np.sin(x)**2 / x**2)

    A = (np.pi * f4**2 / (f4 - f3 + 1e-6)) * _sinc2(f4, t)
    B = (np.pi * f3**2 / (f4 - f3 + 1e-6)) * _sinc2(f3, t)
    C = (np.pi * f2**2 / (f2 - f1 + 1e-6)) * _sinc2(f2, t)
    D = (np.pi * f1**2 / (f2 - f1 + 1e-6)) * _sinc2(f1, t)
    w = (A - B) - (C - D)
    return w / (np.max(np.abs(w)) + 1e-12)


WAVELET_PRESETS = {
    'ricker_25'  : lambda dt: ricker(25,  dt),
    'ricker_40'  : lambda dt: ricker(40,  dt),
    'ricker_60'  : lambda dt: ricker(60,  dt),
    'ormsby_broadband': lambda dt: ormsby(5, 10, 60, 70, dt),
}


# ─────────────────────────────────────────────────────────────────────────────
# DEPTH → TWT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_twt(depth: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """
    Convert depth samples to two-way travel time (TWT) using the
    interval velocity log.

    TWT[0] = 0, units: seconds.
    """
    dz   = np.diff(depth, prepend=depth[0])
    dt_i = 2.0 * np.abs(dz) / (vp + 1e-12)
    return np.cumsum(dt_i) - dt_i[0]


def resample_to_time(signal: np.ndarray, twt_in: np.ndarray,
                     dt_out: float = 0.001) -> tuple:
    """
    Resample depth-domain signal onto uniform time grid.

    Returns
    -------
    (t_uniform, signal_resampled)
    """
    t_out = np.arange(twt_in[0], twt_in[-1], dt_out)
    s_out = np.interp(t_out, twt_in, signal)
    return t_out, s_out


# ─────────────────────────────────────────────────────────────────────────────
# REFLECTION COEFFICIENTS
# ─────────────────────────────────────────────────────────────────────────────

def zero_offset_rc(ai: np.ndarray) -> np.ndarray:
    """Normal-incidence RC from AI log: R[i] = (AI[i+1]-AI[i])/(AI[i+1]+AI[i])"""
    rc      = np.zeros(len(ai))
    denom   = ai[1:] + ai[:-1]
    rc[:-1] = np.where(denom > 0,
                       (ai[1:] - ai[:-1]) / denom,
                       0.0)
    return rc


def shuey_2term(vp1, vp2, vs1, vs2, rho1, rho2,
                angles: np.ndarray = None):
    """
    Shuey (1985) 2-term approximation to Zoeppritz equations.

    R(θ) = R₀ + G·sin²(θ)

    where:
        R₀ = Intercept  (zero-offset RC)
        G  = Gradient

    Parameters
    ----------
    vp1, vp2   : P-wave velocities above and below interface (m/s)
    vs1, vs2   : S-wave velocities (m/s)
    rho1, rho2 : densities (g/cc or kg/m³, same units cancel)
    angles     : optional array of angles (degrees) for R(θ)

    Returns
    -------
    R0  : Intercept (scalar or array)
    G   : Gradient  (scalar or array)
    R   : R(θ) matrix [n_interfaces × n_angles] or None
    """
    vp_avg  = 0.5 * (vp1 + vp2)
    vs_avg  = 0.5 * (vs1 + vs2)
    rho_avg = 0.5 * (rho1 + rho2)

    dvp  = vp2  - vp1
    dvs  = vs2  - vs1
    drho = rho2 - rho1

    R0 = 0.5 * (dvp / (vp_avg + 1e-10) + drho / (rho_avg + 1e-10))
    G  = 0.5 * (
        dvp / (vp_avg + 1e-10)
        - 4.0 * (vs_avg / (vp_avg + 1e-10))**2
        * (2.0 * dvs / (vs_avg + 1e-10) + drho / (rho_avg + 1e-10))
    )

    R = None
    if angles is not None:
        sin2 = np.sin(np.deg2rad(angles)) ** 2
        if np.ndim(R0) > 0:
            R = R0[:, None] + G[:, None] * sin2[None, :]
        else:
            R = R0 + G * sin2
    return R0, G, R


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SeismicEngine:
    """
    Builds synthetic seismograms (zero-offset and pre-stack gathers)
    from a rock physics DataFrame.

    Parameters
    ----------
    df           : DataFrame with VP, VS, RHOB, AI (and optionally AI_gas etc.)
    f_dom        : dominant frequency for Ricker wavelet (Hz)
    wavelet_type : 'ricker' or 'ormsby'
    dt_out       : output time sampling (s), default 1 ms
    angles       : array of incidence angles (°) for pre-stack gather
    """

    def __init__(self,
                 df          : pd.DataFrame,
                 f_dom       : float = 40.0,
                 wavelet_type: str   = 'ricker',
                 dt_out      : float = 0.001,
                 angles      : np.ndarray = None):
        self.df           = df
        self.f_dom        = f_dom
        self.wavelet_type = wavelet_type
        self.dt_out       = dt_out
        self.angles       = angles if angles is not None else np.arange(0, 46, 3)
        self._built       = False

    # ── Build TWT axis & wavelet ──────────────────────────────────────────

    def _setup(self):
        vp       = self.df['VP'].values.astype(float)
        depth    = self.df['DEPTH'].values.astype(float)
        self.twt = depth_to_twt(depth, vp)
        self.dt_wav = float(np.median(np.diff(self.twt)))

        if self.wavelet_type == 'ricker':
            self.wavelet = ricker(self.f_dom, self.dt_wav)
        elif self.wavelet_type == 'ormsby':
            self.wavelet = ormsby(5, 10, self.f_dom, self.f_dom + 10, self.dt_wav)
        else:
            self.wavelet = ricker(self.f_dom, self.dt_wav)

        self._built = True

    # ── Zero-offset synthetic ─────────────────────────────────────────────

    def zero_offset(self, ai_col: str = 'AI') -> dict:
        """
        Build zero-offset synthetic from AI log.

        Returns
        -------
        dict: time, synth_depth, synth_time, rc, twt
        """
        if not self._built:
            self._setup()

        ai   = self.df[ai_col].values.astype(float)
        rc   = zero_offset_rc(ai)
        synd = convolve(rc, self.wavelet, mode='same')
        t, s = resample_to_time(synd, self.twt, self.dt_out)

        return {
            'time'       : t,
            'synth'      : s,
            'synth_depth': synd,
            'rc'         : rc,
            'twt'        : self.twt,
            'wavelet'    : self.wavelet,
            'ai_col'     : ai_col,
        }

    # ── Pre-stack angle gather ────────────────────────────────────────────

    def angle_gather(self, vp_col   : str = 'VP',
                           vs_col   : str = 'VS',
                           rho_col  : str = 'RHOB') -> dict:
        """
        Build pre-stack angle gather using Shuey 2-term.

        Returns
        -------
        dict:
            time    : uniform time axis
            matrix  : ndarray (n_time × n_angles)
            R0      : intercept at each interface
            G       : gradient at each interface
            twt_iface : TWT at each interface midpoint
            angles  : angle array used
        """
        if not self._built:
            self._setup()

        df  = self.df
        vp  = df[vp_col].values.astype(float)
        vs  = df[vs_col].values.astype(float)
        rho = df[rho_col].values.astype(float)

        n = len(vp) - 1
        R0_arr = np.zeros(n)
        G_arr  = np.zeros(n)

        # Compute Intercept and Gradient at all interfaces
        R0_arr, G_arr, _ = shuey_2term(
            vp[:-1], vp[1:], vs[:-1], vs[1:], rho[:-1], rho[1:])

        # Build angle traces
        angle_rcs = {}
        for angle in self.angles:
            rc_a = np.zeros(len(vp))
            sin2 = np.sin(np.deg2rad(angle)) ** 2
            rc_a[:-1] = R0_arr + G_arr * sin2
            angle_rcs[angle] = rc_a

        # Convolve each angle and resample to time
        t_ref, _ = resample_to_time(
            convolve(angle_rcs[self.angles[0]], self.wavelet, mode='same'),
            self.twt, self.dt_out)

        matrix = np.zeros((len(t_ref), len(self.angles)))

        for j, angle in enumerate(self.angles):
            synd = convolve(angle_rcs[angle], self.wavelet, mode='same')
            t_u, s_u = resample_to_time(synd, self.twt, self.dt_out)
            n_min = min(len(t_u), len(t_ref))
            matrix[:n_min, j] = s_u[:n_min]

        # TWT at interface midpoints
        twt_iface = 0.5 * (self.twt[:-1] + self.twt[1:])

        return {
            'time'      : t_ref,
            'matrix'    : matrix,
            'R0'        : R0_arr,
            'G'         : G_arr,
            'twt_iface' : twt_iface,
            'angles'    : self.angles.copy(),
            'wavelet'   : self.wavelet,
        }

    # ── AVO attributes on time grid ───────────────────────────────────────

    def avo_attributes(self, gather_result: dict,
                       dt_out: float = None) -> dict:
        """
        Interpolate Intercept and Gradient onto the uniform time grid.

        Parameters
        ----------
        gather_result : output of angle_gather()

        Returns
        -------
        dict: time, R0_time, G_time
        """
        dt = dt_out or self.dt_out
        t0 = gather_result['twt_iface']
        t_u, R0_t = resample_to_time(gather_result['R0'], t0, dt)
        _,   G_t  = resample_to_time(gather_result['G'],  t0, dt)
        return {'time': t_u, 'R0_time': R0_t, 'G_time': G_t}

    # ── Full run ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Convenience: run zero-offset (brine + gas) + angle gather.

        Assumes df has columns: VP, VS, RHOB, AI, AI_gas, VP_gas, VS_gas, RHOB_gas

        Returns
        -------
        dict with keys: zero_brine, zero_gas, gather_brine, gather_gas, avo_attrs
        """
        if not self._built:
            self._setup()

        result = {}

        result['zero_brine']  = self.zero_offset(ai_col='AI')

        if 'AI_gas' in self.df.columns:
            result['zero_gas'] = self.zero_offset(ai_col='AI_gas')
            result['gather_gas'] = self.angle_gather('VP_gas', 'VS_gas', 'RHOB_gas')

        result['gather_brine'] = self.angle_gather('VP', 'VS', 'RHOB')
        result['avo_attrs']    = self.avo_attributes(result['gather_brine'])

        return result
