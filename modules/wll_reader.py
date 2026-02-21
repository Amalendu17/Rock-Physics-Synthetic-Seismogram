"""
modules/wll_reader.py
=====================
OpendTect V6.6 Binary .wll File Reader
F3 Block Multi-Well Rock Physics Framework

Handles:
  - Parsing OpendTect binary .wll files (float32 depth-value pairs)
  - Auto-discovery of wells and logs from a directory
  - Depth-grid merging via linear interpolation
  - Well metadata inventory reporting

Usage:
    from modules.wll_reader import WellLoader
    loader = WellLoader(wll_dir='./wll_files')
    wells = loader.load_all_wells()          # dict: well_name → DataFrame
    df    = loader.load_well('F02-1')
"""

import os
import glob
import struct
import re
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# MNEMONIC → STANDARD COLUMN NAME MAP
# ─────────────────────────────────────────────────────────────────────────────

MNEMONIC_MAP = {
    'RHOB' : 'RHOB',     # Density          g/cc
    'DT'   : 'DT',       # Sonic            us/ft
    'GR'   : 'GR',       # Gamma Ray        API
    'PHI'  : 'PHIT',     # Porosity         fraction
    'IMP'  : 'AI_obs',   # P-Impedance (observed)
    'PVEL' : 'VP',       # P-wave velocity  m/s
    'SVEL' : 'VS',       # S-wave velocity  m/s
    'LITH' : 'LITH',     # Lithology code   (OpendTect)
}

# Log name keywords for fallback mnemonic detection
NAME_KEYWORDS = {
    'density'    : 'RHOB',
    'sonic'      : 'DT',
    'gamma'      : 'GR',
    'porosity'   : 'PHIT',
    'impedance'  : 'AI_obs',
    'vp'         : 'VP',
    'p-vel'      : 'VP',
    'pvel'       : 'VP',
    'vs'         : 'VS',
    's-vel'      : 'VS',
    'svel'       : 'VS',
    'lith'       : 'LITH',
}

# Physical QC bounds
QC_BOUNDS = {
    'VP'  : (1200.0, 6000.0),
    'VS'  : (400.0,  3500.0),
    'RHOB': (1.50,   3.20),
    'GR'  : (0.0,    300.0),
    'PHIT': (0.0,    0.60),
    'DT'  : (40.0,   250.0),
}

DEPTH_STEP_M = 0.15   # output depth grid spacing (metres)


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_wll(filepath: str) -> dict:
    """
    Parse a single OpendTect V6.6 .wll binary file.

    File layout:
        ASCII header (terminated by second '!\\n')
        Binary data: pairs of little-endian float32  →  (depth_m, value)

    Returns
    -------
    dict with keys:
        name, mnemonic, unit, depth_unit,
        dah_range, log_range,
        depth (np.ndarray, float32),
        values (np.ndarray, float32)
    """
    with open(filepath, 'rb') as fh:
        raw = fh.read()

    first_bang  = raw.find(b'!\n')
    second_bang = raw.find(b'!\n', first_bang + 1)
    if second_bang == -1:
        second_bang = first_bang

    header_str = raw[:second_bang].decode('ascii', errors='replace')

    meta = {}
    for line in header_str.splitlines():
        if ':' in line:
            k, _, v = line.partition(':')
            meta[k.strip()] = v.strip()

    data_start = second_bang + 2
    data_bytes  = raw[data_start:]
    n_pairs     = len(data_bytes) // 8

    if n_pairs == 0:
        return None

    floats = struct.unpack(f'<{n_pairs * 2}f', data_bytes[:n_pairs * 8])
    depth  = np.array(floats[0::2], dtype=np.float32)
    values = np.array(floats[1::2], dtype=np.float32)

    # Mask OpendTect undefined (1e30)
    values[np.abs(values) > 1e20] = np.nan

    return {
        'filepath'  : str(filepath),
        'name'      : meta.get('Name',            Path(filepath).stem),
        'mnemonic'  : meta.get('Mnemonic',        ''),
        'unit'      : meta.get('Unit of Measure', ''),
        'depth_unit': meta.get('Depth-Unit',      'Meter'),
        'dah_range' : meta.get('Dah range',       ''),
        'log_range' : meta.get('Log range',       ''),
        'depth'     : depth,
        'values'    : values,
    }


def _resolve_column(mnemonic: str, name: str) -> str:
    """Resolve standard column name from mnemonic or log name."""
    if mnemonic in MNEMONIC_MAP:
        return MNEMONIC_MAP[mnemonic]
    name_lower = name.lower()
    for kw, col in NAME_KEYWORDS.items():
        if kw in name_lower:
            return col
    return name.replace(' ', '_').replace('-', '_').replace('/', '_')


# ─────────────────────────────────────────────────────────────────────────────
# WELL LOADER
# ─────────────────────────────────────────────────────────────────────────────

class WellLoader:
    """
    Loads OpendTect .wll files from a directory.

    Supports:
        - Auto-discovery of all wells present in the directory
        - Per-well log loading and depth-grid merging
        - QC cleaning (spike removal + interpolation + median smoothing)

    Parameters
    ----------
    wll_dir : str
        Directory containing .wll files.
    depth_step : float
        Output depth grid spacing in metres (default 0.15 m).
    verbose : bool
        Print loading progress.
    """

    def __init__(self, wll_dir: str,
                 depth_step: float = DEPTH_STEP_M,
                 verbose: bool = True):
        self.wll_dir    = str(wll_dir)
        self.depth_step = depth_step
        self.verbose    = verbose
        self._inventory = None   # populated by discover_wells()

    # ── Discovery ──────────────────────────────────────────────────────────

    def discover_wells(self) -> dict:
        """
        Scan directory for .wll files and build an inventory:
            { well_name : [ {filepath, col, log} ] }
        """
        all_files = sorted(glob.glob(os.path.join(self.wll_dir, '*.wll')))
        inventory = {}

        for fp in all_files:
            stem = Path(fp).stem   # e.g. '1771696765748_F02-1_1'
            # Extract well name: look for pattern like F02-1, F03-2, F06-1
            m = re.search(r'(F\d{2}-\d+)', stem, re.IGNORECASE)
            well = m.group(1).upper() if m else 'UNKNOWN'

            if well not in inventory:
                inventory[well] = []

            log = parse_wll(fp)
            if log is None:
                continue

            col = _resolve_column(log['mnemonic'], log['name'])
            # Suffix duplicate columns (e.g. two VP logs → VP, VP_2)
            existing_cols = [e['col'] for e in inventory[well]]
            if col in existing_cols:
                col = col + f"_{existing_cols.count(col) + 1}"

            log['col'] = col
            inventory[well].append({'filepath': fp, 'col': col, 'log': log})

        self._inventory = inventory
        return inventory

    def available_wells(self) -> list:
        """Return list of well names found in directory."""
        if self._inventory is None:
            self.discover_wells()
        return sorted(self._inventory.keys())

    def print_inventory(self):
        """Pretty-print the log inventory."""
        if self._inventory is None:
            self.discover_wells()

        print(f"\n{'═'*72}")
        print(f"  WLL INVENTORY  —  {self.wll_dir}")
        print(f"{'═'*72}")
        for well, entries in sorted(self._inventory.items()):
            print(f"\n  ► Well: {well}  ({len(entries)} logs)")
            print(f"    {'Column':<14} {'Log Name':<35} {'Mnem':<6} "
                  f"{'Depth Range':>22}  {'Unit'}")
            print(f"    {'-'*14} {'-'*35} {'-'*6} {'-'*22}  {'-'*12}")
            for e in entries:
                lg = e['log']
                print(f"    {e['col']:<14} {lg['name'][:34]:<35} "
                      f"{lg['mnemonic']:<6} {lg['dah_range']:>22}  {lg['unit']}")
        print(f"\n{'═'*72}\n")

    # ── Single-well merge ───────────────────────────────────────────────────

    def load_well(self, well_name: str,
                  qc: bool = True,
                  smooth_kernel: int = 7) -> pd.DataFrame:
        """
        Load, merge and (optionally) QC-clean all logs for one well.

        Parameters
        ----------
        well_name    : e.g. 'F02-1'
        qc           : apply spike removal + smoothing
        smooth_kernel: median filter window (must be odd)

        Returns
        -------
        pd.DataFrame with DEPTH + one column per log
        """
        if self._inventory is None:
            self.discover_wells()

        # Case-insensitive match
        key = next((k for k in self._inventory if k.upper() == well_name.upper()), None)
        if key is None:
            raise KeyError(f"Well '{well_name}' not found. "
                           f"Available: {self.available_wells()}")

        entries = self._inventory[key]

        if self.verbose:
            print(f"\n  Loading {key}  ({len(entries)} logs) …")

        # Build per-log series, then merge onto common depth grid
        all_d = np.concatenate([e['log']['depth'] for e in entries])
        d_min, d_max = np.nanmin(all_d), np.nanmax(all_d)
        depth_grid = np.arange(d_min, d_max + self.depth_step, self.depth_step)

        df = pd.DataFrame({'DEPTH': depth_grid})

        for e in entries:
            col = e['col']
            lg  = e['log']
            d, v = lg['depth'], lg['values']
            order = np.argsort(d)
            d, v  = d[order], v[order]
            valid = ~np.isnan(v)
            if valid.sum() < 2:
                continue
            interp = np.interp(depth_grid, d[valid], v[valid],
                               left=np.nan, right=np.nan)
            df[col] = interp.astype(np.float32)

            if self.verbose:
                print(f"    ✓ {col:<14} from {lg['name'][:30]:<30} "
                      f"[{v[valid].min():.3g} – {v[valid].max():.3g}] {lg['unit']}")

        if qc:
            df = self._qc_clean(df, smooth_kernel)

        df = self._infer_lithology(df)
        df.attrs['well'] = key
        return df

    def load_all_wells(self, qc: bool = True,
                        smooth_kernel: int = 7) -> dict:
        """
        Load all discovered wells.
        Returns dict:  { well_name : DataFrame }
        """
        if self._inventory is None:
            self.discover_wells()

        wells = {}
        for name in self.available_wells():
            try:
                wells[name] = self.load_well(name, qc=qc,
                                              smooth_kernel=smooth_kernel)
            except Exception as ex:
                print(f"  ⚠  Could not load {name}: {ex}")
        return wells

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _qc_clean(df: pd.DataFrame, smooth_kernel: int = 7) -> pd.DataFrame:
        """Spike removal → interpolation → median smoothing."""
        from scipy.signal import medfilt
        df = df.copy()

        for col, (lo, hi) in QC_BOUNDS.items():
            if col in df.columns:
                bad = (df[col] < lo) | (df[col] > hi)
                df.loc[bad, col] = np.nan

        for col in df.columns:
            if col == 'DEPTH':
                continue
            if df[col].isna().all():
                continue
            df[col] = df[col].interpolate(method='linear').ffill().bfill()

        k = smooth_kernel if smooth_kernel % 2 == 1 else smooth_kernel + 1
        for col in ['VP', 'VS', 'RHOB']:
            if col in df.columns:
                vals = df[col].values
                if not np.isnan(vals).all():
                    df[col] = medfilt(vals, kernel_size=k)
        return df

    @staticmethod
    def _infer_lithology(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign integer LITHOLOGY column (0=shale, 1=sand).
        Priority: LITH log → GR cut-off → default shale.
        """
        df = df.copy()
        if 'LITH' in df.columns:
            raw = df['LITH'].fillna(30)
            # OpendTect: 10=sand, 15=silt, 20=silty shale, 30=shale
            df['LITHOLOGY'] = np.where(raw <= 12, 1, 0).astype(np.int8)
        elif 'GR' in df.columns:
            df['LITHOLOGY'] = np.where(df['GR'] < 55, 1, 0).astype(np.int8)
        else:
            df['LITHOLOGY'] = np.int8(0)
        return df
