# Multi-Strategy Bioprinting Pipeline

**AI-driven layer decomposition for autonomous multi-strategy tissue deposition.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> The unifying contribution: a volumetric wound decoder predicts *what* to deposit at each depth layer, and a strategy router decides *how* — routing deep structural layers to honeycomb scaffold and surface conformal layers to geodesic boundary-parallel paths.

---

## The Idea

Real tissue has layers with different functions:

| Depth | Tissue | Needs | Strategy |
|-------|--------|-------|----------|
| Deep (dermis) | Structural scaffold for cell migration | Porous, strong lattice | **Honeycomb** |
| Middle (basement membrane) | Transition: support + coverage | Lattice core + conformal edge | **Hybrid** |
| Surface (epidermis) | Conformal cell seeding | Boundary-parallel, uniform | **Geodesic** |

Our volumetric decoder (`PolarDecoder3DLayered`) already predicts **layer_amounts** — how much material per depth layer. The multi-strategy router interprets this as a *deposition strategy selection signal*:

```
PolarDecoder3DLayered
        |
        v
  layer_amounts: (64, 4)
  [0.9, 0.8, 0.5, 0.2]  ← fill amount per layer (deep → surface)
        |
        v
  StrategyRouter.classify_layers()
        |
        v
  [HONEYCOMB, HONEYCOMB, HYBRID, GEODESIC]
        |
        v
  Per-layer toolpath generation
        |
        v
  Unified deposition plan → robot execution → closed-loop feedback
```

---

## Architecture

```
 PERCEIVE ─────── VolumetricWoundEncoder3D (8 views → 3D voxel grid)
      |             PolarDecoder3DLayered → boundary + depth + layer_amounts
      v
 DECOMPOSE ────── StrategyRouter classifies each layer
      |             deep: HONEYCOMB | middle: HYBRID | surface: GEODESIC
      v
 PLAN ─────────── Per-layer toolpath generation
      |             Honeycomb: hex grid + TSP (Alberta/Tec)
      |             Geodesic: heat method iso-contours (MIT)
      |             Hybrid: honeycomb core + geodesic perimeter
      v
 EXECUTE ──────── 8-DOF IK + closed-loop depth feedback
      |
      v
 VERIFY ───────── RealSense D405 re-scan → correct next layer
```

---

## Quick Start

```python
from multistrategy.toolpath import StrategyRouter, StrategyConfig
import numpy as np

# Simulated decoder output
num_radii, num_layers = 64, 4
layer_amounts = np.array([0.9, 0.75, 0.45, 0.15])  # deep → surface
depth_profile = np.random.uniform(2.0, 5.0, num_radii)  # mm

# Wound boundary (circular approximation)
angles = np.linspace(0, 2 * np.pi, num_radii, endpoint=False)
boundary = np.column_stack([20 * np.cos(angles), 15 * np.sin(angles)])

# Route layers to strategies
router = StrategyRouter(StrategyConfig(
    structural_threshold=0.6,
    conformal_threshold=0.3,
))

plan = router.plan(
    layer_amounts=np.tile(layer_amounts, (num_radii, 1)),
    depth_profile_mm=depth_profile,
    boundary_points_mm=boundary,
)

print(f"Layers: {plan.total_layers}")
print(f"Strategy breakdown: {plan.strategy_breakdown}")
print(f"Total path: {plan.total_path_length_mm:.0f} mm")
print(f"Estimated print time: {plan.estimated_print_time_s:.0f} s")
print(f"Estimated volume: {plan.estimated_volume_mm3:.1f} mm³")

for layer in plan.layers:
    print(f"  Layer {layer.layer_index}: {layer.strategy.name} "
          f"(fill={layer.fill_amount:.2f}, {layer.path_length_mm:.0f} mm)")
```

---

## Installation

```bash
git clone https://github.com/dianisay/bioprinting-pipeline-multi-strategy-toolpath.git
cd bioprinting-pipeline-multi-strategy-toolpath
pip install -e ".[dev]"
```

---

## Relation to Other Repos

This repo unifies three lines of research:

| Repo | Phase | Contribution | Toolpath |
|------|-------|-------------|----------|
| [bioprinting-pipeline-honeycomb](https://github.com/dianisay/bioprinting-pipeline-honeycomb) | Phase 1 (Tec/Alberta) | CT-style volumetric vision + closed-loop | Honeycomb scaffold |
| [bioprinting-pipeline-livemesh](https://github.com/dianisay/bioprinting-pipeline-livemesh) | Phase 2 (MIT) | Poisson recon + geodesic paths + coverage | Geodesic conformal |
| [geodesic-currents](https://github.com/dianisay/geodesic-currents) | MIT library | Mesh-free implicit geodesics | Boundary-parallel |
| **This repo** | **Unified** | **AI-driven layer decomposition** | **Multi-strategy** |

The novelty is NOT in any single toolpath method, but in the **learned orchestration**: the neural network predicts layer-wise fill amounts that directly drive the strategy selection, creating a biologically-motivated, fully autonomous deposition pipeline.

---

## Why This Matters

Existing bioprinting systems use a single deposition strategy for all layers. But biological tissue is heterogeneous:

- **Dermis** needs porous scaffold (interconnected pores for vascularization)
- **Epidermis** needs uniform conformal coverage (keratinocyte sheet)
- **Basement membrane** needs both

By training the decoder to predict *per-layer fill amounts*, and routing those predictions to the appropriate toolpath generator, we achieve:

1. **Biologically-motivated architecture** — each layer gets the right deposition strategy
2. **Fully autonomous** — no manual layer-by-layer programming
3. **End-to-end differentiable** — the vision system learns what the planner needs
4. **Closed-loop robust** — depth sensor verifies each layer before proceeding

---

## Citation

```bibtex
@phdthesis{roldan2026bioprinting,
    title  = {CNN-Transformer-Based Machine Learning for 3D Motion Planning
              and Control in In-Situ Robotic Bioprinters for Superficial
              Tissue Regeneration},
    author = {Ayala Rold\'an, Diana Paola},
    school = {Tecnol\'ogico de Monterrey},
    year   = {2026},
    note   = {In collaboration with MIT CSAIL (Prof. Justin Solomon)},
}
```

## License

MIT
