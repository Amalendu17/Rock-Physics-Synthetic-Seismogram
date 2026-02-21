"""
modules/plotting.py
===================
Unified Plotting Library
F3 Block Multi-Well Rock Physics Framework

Contains plotting functions for:
    Phase 1 : Log QC overview (multi-track)
    Phase 2 : Elastic parameter tracks + crossplots
    Phase 3 : Gassmann fluid substitution comparison
    Phase 4a: Zero-offset synthetic seismogram
    Phase 4b: Pre-stack angle gather (wiggle + VA)
    Phase 4c: Brine vs Gas difference + AVO attribute sections
    Phase 4d: Wavelet comparison panel
    Phase 4e: Synthetic-to-well tie
    Phase 5 : Multi-well overlay crossplots

All plots use a dark theme consistent with industry seismic workstation style.

Usage:
    from modules.plotting import (
        plot_log_overview, plot_elastic_params,
        plot_fluid_substitution, plot_zero_offset,
        plot_angle_gather, plot_difference_avo,
        plot_wavelet_comparison, plot_well_tie,
        plot_multi_well_crossplot
    )
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BG_COLOR   = '#ffffff'
PANEL_BG   = '#f7f8fa'
GRID_COLOR = '#d0d5dd'
TEXT_COLOR = '#1a1f2e'

# Per-well colour palette (up to 6 wells) — vivid but readable on white
WELL_COLORS = {
    'F02-1': '#0077b6',   # deep blue
    'F03-2': '#2d9e3a',   # forest green
    'F03-4': '#e07b00',   # burnt orange
    'F06-1': '#c0392b',   # brick red
    'W5'   : '#7b2d8b',   # purple
    'W6'   : '#0e7c7b',   # teal
}
DEFAULT_WELL_COLOR = '#333333'

LITH_COLORS = {0: '#9e8a72', 1: '#f0c040'}
LITH_LABELS = {0: 'Shale / Silt', 1: 'Sand'}

FLUID_COLORS = {
    'brine': '#0077b6',
    'gas'  : '#c0392b',
    'oil'  : '#e07b00',
}

LOG_META = {
    'VP'  : ('Vp (m/s)',         '#0077b6'),
    'VS'  : ('Vs (m/s)',         '#2d9e3a'),
    'RHOB': ('Density (g/cc)',   '#c0392b'),
    'GR'  : ('GR (API)',         '#e07b00'),
    'PHIT': ('Porosity (frac)',  '#0e7c7b'),
    'DT'  : ('DT (us/ft)',       '#7b2d8b'),
    'AI'  : ('AI (10³ kg/m²s)', '#0077b6'),
    'VPVS': ('Vp/Vs',            '#c0392b'),
    'PR'  : ("Poisson's Ratio",  '#2d9e3a'),
    'K_SAT':('K_sat (GPa)',      '#7b2d8b'),
    'G_SAT':('G_sat (GPa)',      '#e07b00'),
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _dark_fig(nrows=1, ncols=1, figsize=(14, 10), suptitle='', sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    fig.patch.set_facecolor(BG_COLOR)
    if suptitle:
        fig.suptitle(suptitle, color=TEXT_COLOR, fontsize=12,
                     fontweight='bold', y=1.01)
    return fig, axes


def _style_ax(ax, title='', xlabel='', ylabel='', grid=True):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title,  color=TEXT_COLOR, fontsize=9, fontweight='bold')
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(GRID_COLOR)
    if grid:
        ax.grid(True, color=GRID_COLOR, alpha=0.7, linestyle='--', linewidth=0.5)


def _lith_colors_arr(lith: np.ndarray) -> list:
    return [LITH_COLORS.get(int(l), '#888888') for l in lith]


def _well_color(well_name: str) -> str:
    return WELL_COLORS.get(well_name, DEFAULT_WELL_COLOR)


def _save(fig, path, dpi=160):
    if path:
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  → {Path(path).name}")
    plt.close()


def wiggle_fill(ax, trace, time, x_offset=0.0, scale=1.0,
                pos_color='#0077b6', neg_color='#c0392b', lw=0.6):
    """Single wiggle trace with pos/neg fill."""
    t = trace * scale + x_offset
    ax.plot(t, time, color='#1a1f2e', lw=lw, zorder=3, alpha=0.9)
    ax.fill_betweenx(time, x_offset, t, where=t > x_offset,
                     color=pos_color, alpha=0.60, zorder=2)
    ax.fill_betweenx(time, x_offset, t, where=t < x_offset,
                     color=neg_color, alpha=0.45, zorder=2)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — LOG QC OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_log_overview(df: pd.DataFrame, well_name: str = '',
                      save_path: str = None):
    """Multi-track log QC overview coloured by lithology."""
    depth  = df['DEPTH'].values
    lith   = df.get('LITHOLOGY', pd.Series(np.zeros(len(df)))).values.astype(int)
    c_arr  = np.array(_lith_colors_arr(lith))
    wcolor = _well_color(well_name)

    tracks = [c for c in ['VP', 'VS', 'RHOB', 'GR', 'PHIT', 'DT']
              if c in df.columns]
    n = len(tracks) + 1

    fig, axes = _dark_fig(1, n, figsize=(2.8*n, 12), sharey=True,
                          suptitle=f'Well Log Overview  —  {well_name}  '
                                   f'[F3 Block, North Sea]')

    # Lithology track
    ax = axes[0] if n > 1 else axes
    for lv, col in LITH_COLORS.items():
        m = lith == lv
        if m.any():
            ax.scatter(np.ones(m.sum()), depth[m], s=10,
                       color=col, marker='s', label=LITH_LABELS[lv])
    _style_ax(ax, title='Lith', ylabel='Depth (m)', grid=False)
    ax.set_xticks([])
    ax.legend(fontsize=6, loc='lower right', facecolor=PANEL_BG,
              labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
    ax.invert_yaxis()

    ax_list = axes[1:] if hasattr(axes, '__len__') else [axes]
    for ax, col in zip(ax_list, tracks):
        lbl, clr = LOG_META.get(col, (col, wcolor))
        ax.scatter(df[col], depth, c=c_arr, s=1.2, alpha=0.35, zorder=1)
        ax.plot(df[col], depth, color=clr, lw=0.9, alpha=0.85, zorder=2)
        _style_ax(ax, title=col, xlabel=lbl)
        ax.set_yticklabels([])

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — ELASTIC PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_elastic_params(df: pd.DataFrame, well_name: str = '',
                        save_path: str = None):
    """Elastic parameter multi-track."""
    depth = df['DEPTH'].values
    lith  = df.get('LITHOLOGY', pd.Series(np.zeros(len(df)))).values.astype(int)
    c_arr = np.array(_lith_colors_arr(lith))

    params = [(c, *LOG_META[c]) for c in ['AI','VPVS','PR','K_SAT','G_SAT']
              if c in df.columns]
    n      = len(params)

    fig, axes = _dark_fig(1, n, figsize=(3.5*n, 12), sharey=True,
                          suptitle=f'Elastic Parameters  —  {well_name}')

    ax_list = axes if hasattr(axes, '__len__') else [axes]
    for i, (ax, (col, lbl, clr)) in enumerate(zip(ax_list, params)):
        ax.scatter(df[col], depth, c=c_arr, s=1.5, alpha=0.35, zorder=1)
        ax.plot(df[col], depth, color=clr, lw=1.0, alpha=0.9, zorder=2)
        _style_ax(ax, title=col, xlabel=lbl)
        if i == 0:
            ax.set_ylabel('Depth (m)', color=TEXT_COLOR, fontsize=8)
            ax.invert_yaxis()
        else:
            ax.set_yticklabels([])

    handles = [Patch(color=c, label=l) for l, c in
               zip(LITH_LABELS.values(), LITH_COLORS.values())]
    ax_list[-1].legend(handles=handles, fontsize=7, loc='lower right',
                       facecolor=PANEL_BG, labelcolor=TEXT_COLOR,
                       edgecolor=GRID_COLOR)

    _save(fig, save_path)


def plot_crossplots(df: pd.DataFrame, well_name: str = '',
                    save_path: str = None):
    """AI vs Vp/Vs and AI vs Poisson's ratio crossplots."""
    lith  = df.get('LITHOLOGY', pd.Series(np.zeros(len(df)))).values.astype(int)
    c_arr = np.array(_lith_colors_arr(lith))

    fig, axes = _dark_fig(1, 2, figsize=(13, 6),
                          suptitle=f'Rock Physics Crossplots  —  {well_name}')

    for ax, (xcol, ycol, xl, yl) in zip(axes, [
        ('AI', 'VPVS', 'AI (10³ kg/m²s)', 'Vp/Vs'),
        ('AI', 'PR',   'AI (10³ kg/m²s)', "Poisson's Ratio"),
    ]):
        ax.scatter(df[xcol], df[ycol], c=c_arr, s=4, alpha=0.5)
        _style_ax(ax, xlabel=xl, ylabel=yl)

    handles = [Patch(color=c, label=l) for l, c in
               zip(LITH_LABELS.values(), LITH_COLORS.values())]
    axes[1].legend(handles=handles, fontsize=8, facecolor=PANEL_BG,
                   labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FLUID SUBSTITUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_fluid_substitution(df: pd.DataFrame, well_name: str = '',
                             tag: str = 'gas', save_path: str = None):
    """Brine vs substituted fluid comparison tracks."""
    depth = df['DEPTH'].values
    pairs = [(c, f'{c}_{tag}', *LOG_META.get(c, (c, TEXT_COLOR))[:1])
             for c in ['VP','AI','VPVS','PR']
             if c in df.columns and f'{c}_{tag}' in df.columns]

    fig, axes = _dark_fig(1, len(pairs), figsize=(4*len(pairs), 12), sharey=True,
                          suptitle=f'Gassmann Fluid Substitution: Brine → {tag.capitalize()}  '
                                   f'—  {well_name}')
    ax_list = axes if hasattr(axes, '__len__') else [axes]

    for i, (ax, (c_orig, c_new, lbl)) in enumerate(zip(ax_list, pairs)):
        ax.plot(df[c_orig], depth, color=FLUID_COLORS['brine'],
                lw=1.3, label='Brine (orig.)')
        ax.plot(df[c_new],  depth, color=FLUID_COLORS.get(tag, '#ef5350'),
                lw=1.3, ls='--', label=f'{tag.capitalize()} (Gassmann)')
        _style_ax(ax, title=c_orig, xlabel=lbl)
        if i == 0:
            ax.set_ylabel('Depth (m)', color=TEXT_COLOR, fontsize=8)
            ax.invert_yaxis()
            ax.legend(fontsize=7, facecolor=PANEL_BG,
                      labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
        else:
            ax.set_yticklabels([])

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4a — ZERO-OFFSET SYNTHETIC
# ─────────────────────────────────────────────────────────────────────────────

def plot_zero_offset(df: pd.DataFrame,
                     zero_brine: dict, zero_gas: dict,
                     well_name: str = '',
                     save_path: str = None):
    """Wavelet | Lith track | Brine synthetic | Gas synthetic."""
    t_b  = zero_brine['time'];  s_b = zero_brine['synth']
    t_g  = zero_gas['time']  ;  s_g = zero_gas['synth']
    wav  = zero_brine['wavelet']
    twt  = zero_brine['twt']

    lith   = df.get('LITHOLOGY', pd.Series(np.zeros(len(df)))).values.astype(int)
    lith_t = np.interp(t_b, twt, lith)

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'Zero-Offset Synthetic Seismogram  —  {well_name}\n'
                 f'F3 Block, North Sea  |  Ricker 40 Hz',
                 color=TEXT_COLOR, fontsize=12, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04,
                           width_ratios=[0.5, 0.5, 2, 2])

    # Wavelet
    ax0 = fig.add_subplot(gs[0])
    dt_wav = float(np.median(np.diff(twt)))
    t_wav  = np.linspace(-len(wav)//2 * dt_wav * 1000,
                           len(wav)//2 * dt_wav * 1000, len(wav))
    ax0.plot(t_wav, wav, color=_well_color(well_name), lw=2)
    ax0.axhline(0, color=GRID_COLOR, lw=0.5)
    _style_ax(ax0, title='Wavelet\n(Ricker 40Hz)', xlabel='ms', ylabel='Amplitude')

    # Lith track
    ax1 = fig.add_subplot(gs[1])
    for lv, col in LITH_COLORS.items():
        m = lith_t.astype(int) == lv
        if m.any():
            ax1.scatter(np.zeros(m.sum()), t_b[m], s=12, color=col,
                        marker='s', label=LITH_LABELS[lv])
    _style_ax(ax1, title='Lith', ylabel='TWT (s)', grid=False)
    ax1.set_xlim(-0.5, 0.5); ax1.set_xticks([])
    ax1.legend(fontsize=6, loc='lower right', facecolor=PANEL_BG,
               labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
    ax1.set_ylim(t_b[-1], t_b[0])

    # Brine synthetic
    ax2 = fig.add_subplot(gs[2])
    sc = np.max(np.abs(s_b)) * 0.95
    wiggle_fill(ax2, s_b / sc * 0.015, t_b, x_offset=0,
                pos_color=FLUID_COLORS['brine'], neg_color='#ef5350')
    ax2.set_xlim(-0.025, 0.025)
    ax2.axvline(0, color=GRID_COLOR, lw=0.4, ls='--', alpha=0.4)
    _style_ax(ax2, title='Brine Case\n(Zero-Offset)', xlabel='Amplitude')
    ax2.set_ylim(t_b[-1], t_b[0]); ax2.set_yticklabels([])

    # Gas synthetic
    ax3 = fig.add_subplot(gs[3])
    sc = np.max(np.abs(s_g)) * 0.95
    wiggle_fill(ax3, s_g / sc * 0.015, t_g, x_offset=0,
                pos_color=FLUID_COLORS['brine'], neg_color='#ef5350')
    ax3.set_xlim(-0.025, 0.025)
    ax3.axvline(0, color=GRID_COLOR, lw=0.4, ls='--', alpha=0.4)
    _style_ax(ax3, title='Gas Case\n(Gassmann Sub.)', xlabel='Amplitude')
    ax3.set_ylim(t_g[-1], t_g[0]); ax3.set_yticklabels([])

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4b — PRE-STACK ANGLE GATHER
# ─────────────────────────────────────────────────────────────────────────────

def plot_angle_gather(gather: dict, well_name: str = '',
                      case: str = 'brine', save_path: str = None):
    """Pre-stack angle gather: wiggle display + variable density (VA)."""
    t       = gather['time']
    M       = gather['matrix']
    angles  = gather['angles']

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'Pre-Stack Angle Gather ({case.capitalize()})  —  {well_name}\n'
                 f'Angles 0–45°  |  Ricker 40 Hz',
                 color=TEXT_COLOR, fontsize=12, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06, width_ratios=[3, 2])

    # Wiggle
    ax_w = fig.add_subplot(gs[0])
    ax_w.set_facecolor(PANEL_BG)
    amp_scale = np.percentile(np.abs(M), 95) + 1e-12
    wcolor = FLUID_COLORS.get(case, _well_color(well_name))

    for j, angle in enumerate(angles):
        trace = M[:, j] / amp_scale
        wiggle_fill(ax_w, trace, t, x_offset=float(angle),
                    scale=0.85, pos_color=wcolor, neg_color='#ef5350', lw=0.4)

    for a_mark in [0, 15, 30, 45]:
        if a_mark <= angles[-1]:
            ax_w.axvline(a_mark, color=GRID_COLOR, lw=0.5, ls='--', alpha=0.5)
            ax_w.text(a_mark, t[0], f'{a_mark}°', color=TEXT_COLOR,
                      fontsize=7, ha='center', va='bottom')

    ax_w.set_xlim(angles[0] - 2, angles[-1] + 2)
    ax_w.set_ylim(t[-1], t[0])
    _style_ax(ax_w, title='Wiggle Display',
              xlabel='Angle of Incidence (°)', ylabel='TWT (s)')

    # VA image
    ax_v = fig.add_subplot(gs[1])
    ax_v.set_facecolor(PANEL_BG)
    vmax = np.percentile(np.abs(M), 98)
    im = ax_v.imshow(M, aspect='auto', cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax,
                      extent=[angles[0], angles[-1], t[-1], t[0]],
                      interpolation='bilinear')
    _style_ax(ax_v, title='Variable Density (VA)',
              xlabel='Angle (°)')
    ax_v.set_yticklabels([])
    cbar = plt.colorbar(im, ax=ax_v, fraction=0.03, pad=0.01)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cbar.set_label('Amplitude', color=TEXT_COLOR, fontsize=8)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4c — DIFFERENCE + AVO ATTRIBUTE SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_difference_avo(gather_brine: dict, gather_gas: dict,
                         avo_attrs: dict,
                         well_name: str = '',
                         save_path: str = None):
    """Brine | Gas | Difference gathers + R0 trace + G trace."""
    t_b  = gather_brine['time'];  Mb = gather_brine['matrix']
    t_g  = gather_gas['time']  ;  Mg = gather_gas['matrix']
    angles = gather_brine['angles']

    n_min = min(Mb.shape[0], Mg.shape[0])
    Mb = Mb[:n_min]; Mg = Mg[:n_min]
    t  = t_b[:n_min]
    Md = Mg - Mb

    t_avo = avo_attrs['time']
    R0_t  = avo_attrs['R0_time']
    G_t   = avo_attrs['G_time']

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'Brine vs Gas Gather  |  AVO Intercept & Gradient  —  {well_name}',
                 color=TEXT_COLOR, fontsize=12, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.04,
                           width_ratios=[2, 2, 2, 1.1, 1.1])

    def _va(ax, M, title, cmap='RdBu_r'):
        vmax = np.percentile(np.abs(M), 98) + 1e-12
        im   = ax.imshow(M, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
                          extent=[angles[0], angles[-1], t[-1], t[0]],
                          interpolation='bilinear')
        ax.set_facecolor(PANEL_BG)
        _style_ax(ax, title=title, xlabel='Angle (°)', grid=False)
        return im

    ax0 = fig.add_subplot(gs[0]); _va(ax0, Mb, 'Brine Gather')
    ax0.set_ylabel('TWT (s)', color=TEXT_COLOR, fontsize=9)

    ax1 = fig.add_subplot(gs[1]); _va(ax1, Mg, 'Gas Gather (Gassmann)')
    ax1.set_yticklabels([])

    ax2 = fig.add_subplot(gs[2])
    im_d = _va(ax2, Md, 'Difference (Gas − Brine)', cmap='PuOr')
    ax2.set_yticklabels([])
    cbar = plt.colorbar(im_d, ax=ax2, fraction=0.04, pad=0.01)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cbar.set_label('ΔAmplitude', color=TEXT_COLOR, fontsize=7)

    for ax, data, title, pos_col in [
        (fig.add_subplot(gs[3]), R0_t, 'Intercept R₀', '#0077b6'),
        (fig.add_subplot(gs[4]), G_t,  'Gradient G',    '#2d9e3a'),
    ]:
        ax.set_facecolor(PANEL_BG)
        ax.plot(data, t_avo, color=pos_col, lw=1.2)
        ax.fill_betweenx(t_avo, 0, data, where=data > 0, color=pos_col, alpha=0.5)
        ax.fill_betweenx(t_avo, 0, data, where=data < 0, color='#ef5350', alpha=0.5)
        ax.axvline(0, color=GRID_COLOR, lw=0.5, ls='--', alpha=0.5)
        ax.set_ylim(t_avo[-1], t_avo[0])
        _style_ax(ax, title=title, xlabel=title)
        ax.set_yticklabels([])

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4d — WAVELET COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_wavelet_comparison(df: pd.DataFrame,
                             engine_cls,      # SeismicEngine class
                             well_name: str = '',
                             save_path: str = None):
    """Sensitivity study: synthetics with Ricker 25/40/60 Hz + Ormsby."""
    from modules.seismic_engine import ricker, ormsby, depth_to_twt, resample_to_time
    from modules.seismic_engine import zero_offset_rc
    from scipy.signal import convolve

    vp  = df['VP'].values.astype(float)
    twt = depth_to_twt(df['DEPTH'].values.astype(float), vp)
    dt_wav = float(np.median(np.diff(twt)))
    ai     = df['AI'].values.astype(float)
    rc     = zero_offset_rc(ai)

    wavelet_specs = [
        ('Ricker 25 Hz',          ricker(25, dt_wav),        '#0077b6'),
        ('Ricker 40 Hz',          ricker(40, dt_wav),        '#e07b00'),
        ('Ricker 60 Hz',          ricker(60, dt_wav),        '#2d9e3a'),
        ('Ormsby 5–10–60–70 Hz',  ormsby(5,10,60,70,dt_wav), '#7b2d8b'),
    ]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'Wavelet Sensitivity Study  —  {well_name}  (Brine Case)',
                 color=TEXT_COLOR, fontsize=12, fontweight='bold', y=1.01)

    n   = len(wavelet_specs)
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.06, wspace=0.04,
                             height_ratios=[1, 5])

    for j, (name, wav, clr) in enumerate(wavelet_specs):
        synd = convolve(rc, wav, mode='same')
        t_u, s_u = resample_to_time(synd, twt, 0.001)

        t_w = np.linspace(-len(wav)//2 * dt_wav * 1000,
                           len(wav)//2 * dt_wav * 1000, len(wav))

        ax_w = fig.add_subplot(gs[0, j])
        ax_w.plot(t_w, wav, color=clr, lw=1.8)
        ax_w.axhline(0, color=GRID_COLOR, lw=0.4)
        ax_w.set_facecolor(PANEL_BG)
        ax_w.set_title(name, color=TEXT_COLOR, fontsize=8, fontweight='bold')
        ax_w.set_xlabel('ms', color=TEXT_COLOR, fontsize=6)
        ax_w.tick_params(colors=TEXT_COLOR, labelsize=6)
        for sp in ax_w.spines.values():
            sp.set_color(GRID_COLOR)
        if j > 0:
            ax_w.set_yticklabels([])

        ax_s = fig.add_subplot(gs[1, j])
        ax_s.set_facecolor(PANEL_BG)
        sc = np.max(np.abs(s_u)) + 1e-12
        wiggle_fill(ax_s, s_u / sc * 0.018, t_u, x_offset=0,
                    pos_color=clr, neg_color='#ef5350', lw=0.5)
        ax_s.set_xlim(-0.03, 0.03)
        ax_s.set_ylim(t_u[-1], t_u[0])
        ax_s.axvline(0, color=GRID_COLOR, lw=0.4, ls='--', alpha=0.4)
        ax_s.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax_s.set_xlabel('Amplitude', color=TEXT_COLOR, fontsize=8)
        for sp in ax_s.spines.values():
            sp.set_color(GRID_COLOR)
        if j == 0:
            ax_s.set_ylabel('TWT (s)', color=TEXT_COLOR, fontsize=9)
        else:
            ax_s.set_yticklabels([])

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4e — WELL TIE PANEL
# ─────────────────────────────────────────────────────────────────────────────

def plot_well_tie(df: pd.DataFrame, zero_brine: dict, zero_gas: dict,
                   well_name: str = '', save_path: str = None):
    """GR → AI brine → AI gas → RC → Brine syn → Gas syn."""
    from modules.seismic_engine import depth_to_twt, resample_to_time, zero_offset_rc

    twt   = zero_brine['twt']
    t_b   = zero_brine['time'];  s_b = zero_brine['synth']
    t_g   = zero_gas['time']  ;  s_g = zero_gas['synth']

    rc    = zero_offset_rc(df['AI'].values.astype(float))
    t_rc, s_rc = resample_to_time(rc, twt, 0.001)
    t_ai, ai_t  = resample_to_time(df['AI'].values.astype(float), twt, 0.001)
    _,  aig_t  = resample_to_time(df['AI_gas'].values.astype(float), twt, 0.001)

    has_gr = 'GR' in df.columns
    panels = []
    if has_gr:
        _, gr_t = resample_to_time(df['GR'].values.astype(float), twt, 0.001)
        panels.append(('GR', gr_t, '#e07b00', False))
    panels += [
        ('AI Brine',   ai_t,  FLUID_COLORS['brine'],  False),
        ('AI Gas',     aig_t, FLUID_COLORS['gas'],    False),
        ('RC',         s_rc,  TEXT_COLOR,              True),
        ('Brine Syn',  s_b,   FLUID_COLORS['brine'],  True),
        ('Gas Syn',    s_g,   FLUID_COLORS['gas'],    True),
    ]

    fig = plt.figure(figsize=(3.2*len(panels), 13))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'Synthetic-to-Well Tie  —  {well_name}',
                 color=TEXT_COLOR, fontsize=12, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(1, len(panels), figure=fig, wspace=0.04)

    for i, (title, data, clr, is_wig) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor(PANEL_BG)
        t_ref = t_b if is_wig else t_ai

        if is_wig:
            sc = np.max(np.abs(data)) + 1e-12
            wiggle_fill(ax, data / sc * 0.012, t_ref, x_offset=0,
                        pos_color=clr, neg_color='#ef5350', lw=0.8)
            ax.set_xlim(-0.02, 0.02)
            ax.axvline(0, color=GRID_COLOR, lw=0.4, ls='--', alpha=0.4)
        else:
            ax.plot(data, t_ref, color=clr, lw=1.0)
            if title == 'AI Gas':
                ax.plot(ai_t, t_ref, color=FLUID_COLORS['brine'], lw=0.6, alpha=0.35)

        ax.set_ylim(t_ref[-1], t_ref[0])
        _style_ax(ax, title=title, xlabel=title)
        if i == 0:
            ax.set_ylabel('TWT (s)', color=TEXT_COLOR, fontsize=9)
        else:
            ax.set_yticklabels([])

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — MULTI-WELL OVERLAY CROSSPLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_well_crossplot(wells: dict, save_path: str = None):
    """
    AI vs Vp/Vs crossplot with all wells overlaid (brine + gas).

    Parameters
    ----------
    wells : dict  { well_name : DataFrame }
            Each DataFrame must have AI, VPVS, AI_gas, VPVS_gas columns.
    """
    fig, axes = _dark_fig(1, 2, figsize=(15, 7),
                          suptitle='Multi-Well AI vs Vp/Vs Crossplot  '
                                   '—  F3 Block, North Sea')

    for ax, (xcol, ycol, title) in zip(axes, [
        ('AI',    'VPVS',    'Brine Case'),
        ('AI_gas','VPVS_gas','Gas-Substituted'),
    ]):
        for wname, df in wells.items():
            if xcol not in df.columns or ycol not in df.columns:
                continue
            wc = _well_color(wname)
            src = df.attrs.get('source', 'real')
            lw  = 0.5 if src == 'synthetic' else 0.5
            ax.scatter(df[xcol], df[ycol],
                       s=3, alpha=0.35, color=wc,
                       label=f'{wname} ({"syn" if src=="synthetic" else "real"})')
        _style_ax(ax, title=title,
                  xlabel='AI (10³ kg/m²s)', ylabel='Vp/Vs')
        ax.legend(fontsize=8, facecolor=PANEL_BG,
                  labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR,
                  markerscale=3)

    _save(fig, save_path)


def plot_multi_well_logs(wells: dict, log_col: str = 'VP',
                          save_path: str = None):
    """
    Single-track overlay of one log curve across all wells.

    Parameters
    ----------
    wells   : dict { well_name : DataFrame }
    log_col : column to plot (e.g. 'VP', 'GR', 'AI')
    """
    lbl, _ = LOG_META.get(log_col, (log_col, TEXT_COLOR))

    fig, ax = _dark_fig(1, 1, figsize=(7, 13),
                         suptitle=f'Multi-Well {log_col} Overlay  —  F3 Block')
    ax = ax if not hasattr(ax, '__len__') else ax[0]

    for wname, df in wells.items():
        if log_col not in df.columns:
            continue
        ax.plot(df[log_col], df['DEPTH'],
                color=_well_color(wname), lw=1.0, alpha=0.85,
                label=wname + (' (syn)' if df.attrs.get('source')=='synthetic' else ''))

    _style_ax(ax, title=f'{log_col} — All Wells',
              xlabel=lbl, ylabel='Depth (m)')
    ax.invert_yaxis()
    ax.legend(fontsize=9, facecolor=PANEL_BG,
              labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)

    _save(fig, save_path)
