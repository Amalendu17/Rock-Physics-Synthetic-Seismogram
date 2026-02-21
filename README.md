# 🌊 F3 Block — Rock Physics & Synthetic Seismogram Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-signal%20processing-8CAAE6?logo=scipy)](https://scipy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Automated multi-well rock physics modelling and synthetic seismogram generation for the F3 Block, North Sea.**  
> Reads real OpendTect `.wll` binary logs for all four wells (F02-1, F03-2, F03-4, F06-1) and runs a full 5-phase workflow producing 44 publication-quality figures and CSV exports.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow Phases](#workflow-phases)
- [Output Figures](#output-figures)
- [Adding a New Well](#adding-a-new-well)
- [Rock Physics Theory](#rock-physics-theory)
- [F3 Block Geology](#f3-block-geology)
- [References](#references)

---

## Overview

This framework provides an end-to-end, zero-manual-intervention pipeline for:

| Capability | Details |
|---|---|
| **WLL Parsing** | Reads OpendTect V6.6 binary `.wll` files; auto-discovers wells by regex; merges logs to 0.15 m depth grid |
| **Log QC** | Spike removal (physical bounds), linear interpolation, median filter smoothing |
| **Elastic Params** | AI, SI, Vp/Vs, Poisson's ratio, K_sat, G_sat from Vp + Vs + RHOB |
| **Gassmann Sub.** | Full 3-step brine → gas substitution; fluid and mineral property databases |
| **Synthetic Seismic** | Zero-offset + pre-stack 0–45° gathers; Ricker & Ormsby wavelets in TWT domain |
| **AVO Analysis** | Shuey 2-term intercept–gradient; difference sections; R₀/G attribute traces |
| **Multi-Well** | Cross-well AI vs Vp/Vs crossplots; log overlay figures for all wells |

---

## Project Structure

```
f3_rockphysics/
├── run_pipeline.py              ← Single entry point, CLI orchestrator
├── modules/
│   ├── wll_reader.py            ← OpendTect .wll parser + WellLoader class
│   ├── rock_physics.py          ← Elastic params + Gassmann substitution
│   ├── seismic_engine.py        ← Wavelets, depth→TWT, RC, angle gathers
│   └── plotting.py              ← All 9 figure types, light/white theme
├── wll_files/                   ← Drop .wll files here (auto-discovered)
└── outputs/                     ← Generated figures and CSVs
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `wll_reader.py` | Parses binary OpendTect files; auto-discovers wells; QC-cleans with spike removal + smoothing |
| `rock_physics.py` | Phase 2 elastic params + Phase 3 full Gassmann with fluid/mineral databases |
| `seismic_engine.py` | `SeismicEngine` class: depth→TWT, Ricker/Ormsby wavelets, Shuey 2-term AVO gathers |
| `plotting.py` | 9 plot functions, light/white theme, multi-well overlay support |

---

## Installation

**Requirements:** Python 3.10+

```bash
pip install numpy scipy pandas matplotlib
```

```bash
git clone https://github.com/Amalendu17/Rock-Physics-Synthetic-Seismogram.git
cd Rock-Physics-Synthetic-Seismogram
```

---

## Quick Start

```bash
# Run all 4 wells, all 5 phases
python run_pipeline.py --wll_dir ./wll_files --out_dir ./outputs

# Skip seismic phase (rock physics only)
python run_pipeline.py --skip_phases 4

# Only multi-well overlay figures
python run_pipeline.py --only_phases 5

# Single well
python run_pipeline.py --wells F02-1
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--wll_dir` | `./wll_files` | Directory containing `.wll` files |
| `--out_dir` | `./outputs` | Root output directory |
| `--wells` | `F02-1 F03-2 F03-4 F06-1` | Space-separated list of wells to process |
| `--skip_phases` | *(none)* | Phase numbers to skip (1–5) |
| `--only_phases` | *(none)* | Run only these phases |
| `--f_dom` | `40.0` | Ricker wavelet dominant frequency (Hz) |
| `--no_real` | `False` | Skip loading real `.wll` files |

---

## Workflow Phases

```
Phase 1  →  Data Loading & QC
Phase 2  →  Elastic Parameters
Phase 3  →  Gassmann Fluid Substitution
Phase 4  →  Synthetic Seismograms
Phase 5  →  Multi-Well Overlays
```

| # | Phase | Key Actions | Outputs |
|---|---|---|---|
| **1** | Data Loading & QC | Parse `.wll`; merge to depth grid; spike removal; smoothing; lithology from LITH/GR | `01_log_overview.png` |
| **2** | Elastic Parameters | AI, SI, Vp/Vs, Poisson's ratio, K_sat, G_sat, M_sat | `02_elastic_params.png` `03_crossplots.png` |
| **3** | Gassmann Sub. | Back-strip brine → K_dry → forward gas substitution → recompute velocities + density | `04_fluid_sub.png` |
| **4** | Synthetic Seismogram | Depth→TWT; zero-offset + pre-stack 0–45°; Ricker/Ormsby; AVO R₀/G sections; well tie | `05–09_*.png` |
| **5** | Multi-Well Overlays | AI vs Vp/Vs crossplot (brine + gas); VP/GR/AI/VPVS log overlays | `multi_well/*.png` |

---

## Output Figures

Each well produces **9 figures** plus a `rock_physics.csv`, totalling **44 figures** across all 4 wells:

```
outputs/
├── F02-1/
│   ├── 01_log_overview.png          # Multi-track log QC
│   ├── 02_elastic_params.png        # AI, Vp/Vs, PR, K, G tracks
│   ├── 03_crossplots.png            # AI vs Vp/Vs and Poisson's ratio
│   ├── 04_fluid_sub.png             # Brine vs gas comparison tracks
│   ├── 05_zero_offset.png           # Zero-offset wiggle synthetic
│   ├── 06_gather_brine.png          # Pre-stack gather (wiggle + VA)
│   ├── 06_gather_gas.png            # Pre-stack gather gas case
│   ├── 07_diff_avo.png              # Difference section + R0/G traces
│   ├── 08_wavelet_comparison.png    # Ricker 25/40/60 Hz + Ormsby
│   ├── 09_well_tie.png              # GR→AI→RC→Brine→Gas panel
│   └── rock_physics.csv
├── F03-2/   (same structure)
├── F03-4/   (same structure)
├── F06-1/   (same structure)
└── multi_well/
    ├── crossplot_brine_gas.png      # All wells, AI vs Vp/Vs
    ├── log_overlay_VP.png
    ├── log_overlay_GR.png
    ├── log_overlay_AI.png
    └── log_overlay_VPVS.png
```

### CSV Column Reference

| Column | Units | Description |
|---|---|---|
| `DEPTH` | m | Depth on 0.15 m grid |
| `VP` / `VS` | m/s | P- and S-wave velocities (QC-cleaned) |
| `RHOB` | g/cc | Bulk density |
| `GR` | API | Gamma Ray |
| `PHIT` | fraction | Total porosity |
| `AI` / `SI` | 10³ kg/m²s | Acoustic and shear impedance |
| `VPVS` / `PR` | — | Vp/Vs ratio and Poisson's ratio |
| `K_SAT` / `G_SAT` | GPa | Saturated bulk and shear moduli |
| `VP_gas` / `VS_gas` | m/s | Gas-substituted velocities |
| `AI_gas` / `VPVS_gas` | — | Gas-substituted AI and Vp/Vs |
| `RHOB_gas` | g/cc | Gas-substituted density |
| `K_dry` | GPa | Dry-frame bulk modulus |

---

## Adding a New Well

**Real well (with `.wll` files):**

1. Copy all `.wll` files into `--wll_dir`
2. Filenames must contain the well ID (e.g. `F04-1`) — auto-discovered by regex `F\d{2}-\d`
3. Add the well name to `REAL_WELLS` in `run_pipeline.py`
4. Run: `python run_pipeline.py --wells F02-1 F03-2 F03-4 F06-1 F04-1`

```python
# run_pipeline.py
REAL_WELLS = ['F02-1', 'F03-2', 'F03-4', 'F06-1', 'F04-1']   # ← add here
```

---

## Rock Physics Theory

### Gassmann (1951) Fluid Substitution

3-step workflow:

**Step 1 — Back-strip to dry frame:**
```
K_dry = [K_sat · (φ·K_min/K_fl + 1 − φ) − K_min]
        ─────────────────────────────────────────────
         φ·K_min/K_fl + K_sat/K_min − 1 − φ
```

**Step 2 — Forward Gassmann with new fluid:**
```
K_sat_new = K_dry + (1 − K_dry/K_min)²
                    ────────────────────────────────────
                     φ/K_fl_new + (1−φ)/K_min − K_dry/K_min²
```

**Step 3 — Update density and recompute velocities:**
```
ρ_new  = ρ_dry + φ · ρ_fl_new
Vp_new = √( (K_sat_new + 4G/3) × 10⁹ / ρ_new )
Vs_new = √( G × 10⁹ / ρ_new )          ← fluid-independent
```

> **Assumptions:** Low-frequency (seismic) limit · Homogeneous isotropic monomineralic frame · Fully connected equilibrated pore space · No chemical fluid–frame interaction

### Shuey (1985) 2-Term AVO

```
R(θ) = R₀ + G · sin²(θ)

R₀ = ½ (ΔVp/V̄p + Δρ/ρ̄)                           [Intercept]
G  = ½ [ΔVp/V̄p − 4(V̄s/V̄p)²·(2ΔVs/V̄s + Δρ/ρ̄)]   [Gradient]
```

The F3 Block gas sands show **Class III AVO**: negative intercept R₀ and negative gradient G — the intercept–gradient point falls in Quadrant III.

---

## F3 Block Geology

The F3 Block (Netherlands sector, North Sea) contains Pliocene–Pleistocene unconsolidated sands sealed by Holocene marine clays. The gas sands are characterised by low acoustic impedance relative to the overlying shale.

| Well | Data | Depth Range | AI Change (→ Gas) | Vp/Vs Change |
|---|---|---|---|---|
| **F02-1** | Real `.wll` | 50–1500 m | −40% (4299 → 2596) | −35% (1.82 → 1.19) |
| **F03-2** | Real `.wll` | 30–2140 m | −31% (5437 → 3759) | −28% (2.26 → 1.64) |
| **F03-4** | Real `.wll` | 30–1900 m | −44% (4431 → 2474) | −40% (1.97 → 1.18) |
| **F06-1** | Real `.wll` | 29–1700 m | −39% (4522 → 2764) | −35% (1.86 → 1.21) |

---

## References

- Gassmann, F. (1951). Uber die Elastizitat poroser Medien. *Vierteljahrsschrift der Naturforschenden Gesellschaft in Zurich*, 96, 1–23.
- Shuey, R. T. (1985). A simplification of the Zoeppritz equations. *Geophysics*, 50(4), 609–614.
- Smith, T. M., Sondergeld, C. H., & Rai, C. S. (2003). Gassmann fluid substitutions: A tutorial. *Geophysics*, 68(2), 430–440.
- Mavko, G., Mukerji, T., & Dvorkin, J. (2009). *The Rock Physics Handbook* (2nd ed.). Cambridge University Press.
- dGB Earth Sciences. [OpendTect F3 Demo Dataset](https://opendtect.org/osr/pmwiki.php/Main/F3BlockNetherlandsOpenSeismicDataset)

---

*F3 Block · North Sea · Rock Physics & Seismic Framework · 2026*
