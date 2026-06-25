"""Multi-Strategy Toolpath Router.

The core innovation: the volumetric decoder predicts layer_amounts (B, num_radii, num_layers),
which encodes HOW MUCH material to deposit at each depth layer. This module interprets that
prediction to decide WHICH deposition strategy to use per layer:

    - Deep layers (high structural fill) -> honeycomb scaffold (Alberta/Tec)
    - Surface layers (low conformal fill) -> geodesic boundary-parallel paths (MIT)
    - Transition layers -> hybrid (honeycomb core + geodesic perimeter)

The decision is differentiable w.r.t. the decoder output, enabling end-to-end training
of the full perception-to-deposition pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LayerStrategy(Enum):
    """Deposition strategy for a single layer."""
    HONEYCOMB = auto()    # Structural scaffold (hex grid + TSP)
    GEODESIC = auto()     # Conformal surface coverage (boundary-parallel)
    HYBRID = auto()       # Honeycomb core + geodesic perimeter ring


@dataclass
class StrategyConfig:
    """Configuration for the multi-strategy router."""

    # Threshold: layers with fill_amount > this use honeycomb (structural)
    structural_threshold: float = 0.6

    # Threshold: layers with fill_amount < this use geodesic (conformal)
    conformal_threshold: float = 0.3

    # Between thresholds: hybrid strategy
    # (automatically derived: conformal_threshold < x < structural_threshold)

    # Honeycomb parameters
    hex_cell_size_mm: float = 3.0
    nozzle_diameter_mm: float = 0.4

    # Geodesic parameters
    geodesic_spacing_mm: float = 1.5
    adaptive_curvature: bool = True
    curvature_factor: float = 0.5

    # Hybrid: what fraction of the layer width gets geodesic perimeter
    hybrid_perimeter_fraction: float = 0.25


@dataclass
class LayerPlan:
    """Planned toolpath for a single layer."""
    layer_index: int
    strategy: LayerStrategy
    fill_amount: float
    depth_mm: float
    waypoints_mm: NDArray[np.float64]  # (N, 3)
    normals: NDArray[np.float64]       # (N, 3)
    is_deposition: NDArray[np.bool_]   # (N,) True where extruding
    path_length_mm: float
    num_cells: int = 0                 # for honeycomb layers


@dataclass
class MultiStrategyPlan:
    """Complete multi-strategy deposition plan for a wound."""
    layers: list[LayerPlan]
    total_path_length_mm: float
    total_layers: int
    strategy_breakdown: dict[str, int]  # e.g. {"honeycomb": 2, "geodesic": 1, "hybrid": 1}
    estimated_print_time_s: float
    estimated_volume_mm3: float


class StrategyRouter:
    """Routes each layer to the appropriate deposition strategy.

    The router takes the decoder's layer_amounts prediction and the wound geometry,
    then decides per-layer: honeycomb, geodesic, or hybrid.

    Decision logic:
        layer_amounts[i] > structural_threshold  -> HONEYCOMB (deep, needs scaffold)
        layer_amounts[i] < conformal_threshold   -> GEODESIC (surface, needs coverage)
        otherwise                                -> HYBRID (transition zone)

    This mimics biological tissue architecture:
        - Dermis (deep): needs structural scaffold for fibroblast migration
        - Epidermis (surface): needs conformal coverage for keratinocyte seeding
        - Basement membrane (transition): needs both
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        logger.info(
            f"StrategyRouter initialized: structural>{self.config.structural_threshold}, "
            f"conformal<{self.config.conformal_threshold}"
        )

    def classify_layers(
        self,
        layer_amounts: NDArray[np.float64],
        depth_profile_mm: NDArray[np.float64],
    ) -> list[LayerStrategy]:
        """Classify each layer into a deposition strategy.

        Args:
            layer_amounts: (num_radii, num_layers) predicted fill amounts [0, 1]
            depth_profile_mm: (num_radii,) wound depth at each angle

        Returns:
            List of LayerStrategy, one per layer (deepest first)
        """
        num_layers = layer_amounts.shape[1] if layer_amounts.ndim == 2 else layer_amounts.shape[0]

        # Average fill amount across all radii for each layer
        if layer_amounts.ndim == 2:
            mean_fill_per_layer = layer_amounts.mean(axis=0)  # (num_layers,)
        else:
            mean_fill_per_layer = layer_amounts

        strategies = []
        for i in range(num_layers):
            fill = float(mean_fill_per_layer[i])
            if fill > self.config.structural_threshold:
                strategies.append(LayerStrategy.HONEYCOMB)
            elif fill < self.config.conformal_threshold:
                strategies.append(LayerStrategy.GEODESIC)
            else:
                strategies.append(LayerStrategy.HYBRID)

        logger.info(
            f"Layer classification: {[s.name for s in strategies]} "
            f"(fills: {[f'{f:.2f}' for f in mean_fill_per_layer]})"
        )
        return strategies

    def plan(
        self,
        layer_amounts: NDArray[np.float64],
        depth_profile_mm: NDArray[np.float64],
        boundary_points_mm: NDArray[np.float64],
        surface_mesh=None,
    ) -> MultiStrategyPlan:
        """Generate the full multi-strategy deposition plan.

        Args:
            layer_amounts: (num_radii, num_layers) from decoder
            depth_profile_mm: (num_radii,) wound depth per angle
            boundary_points_mm: (num_radii, 2) wound boundary in mm
            surface_mesh: optional trimesh.Trimesh for geodesic computation

        Returns:
            MultiStrategyPlan with per-layer toolpaths
        """
        strategies = self.classify_layers(layer_amounts, depth_profile_mm)
        num_layers = len(strategies)

        max_depth = float(depth_profile_mm.max())
        layer_height = max_depth / max(num_layers, 1)

        layers = []
        strategy_count = {"honeycomb": 0, "geodesic": 0, "hybrid": 0}

        for i, strategy in enumerate(strategies):
            depth_at_layer = max_depth - i * layer_height
            mean_fill = float(layer_amounts.mean(axis=0)[i]) if layer_amounts.ndim == 2 else float(layer_amounts[i])

            if strategy == LayerStrategy.HONEYCOMB:
                layer_plan = self._plan_honeycomb_layer(
                    i, depth_at_layer, mean_fill, boundary_points_mm
                )
                strategy_count["honeycomb"] += 1
            elif strategy == LayerStrategy.GEODESIC:
                layer_plan = self._plan_geodesic_layer(
                    i, depth_at_layer, mean_fill, boundary_points_mm, surface_mesh
                )
                strategy_count["geodesic"] += 1
            else:
                layer_plan = self._plan_hybrid_layer(
                    i, depth_at_layer, mean_fill, boundary_points_mm, surface_mesh
                )
                strategy_count["hybrid"] += 1

            layers.append(layer_plan)

        total_length = sum(lp.path_length_mm for lp in layers)
        # Estimate: 25 mm/s travel, 10 mm/s deposition
        dep_length = sum(
            float(lp.is_deposition.sum()) / max(len(lp.is_deposition), 1) * lp.path_length_mm
            for lp in layers
        )
        travel_length = total_length - dep_length
        estimated_time = dep_length / 10.0 + travel_length / 25.0

        # Volume estimate: nozzle_diameter * layer_height * deposition_length
        nozzle_area = np.pi * (self.config.nozzle_diameter_mm / 2) ** 2
        estimated_volume = nozzle_area * dep_length

        plan = MultiStrategyPlan(
            layers=layers,
            total_path_length_mm=total_length,
            total_layers=num_layers,
            strategy_breakdown=strategy_count,
            estimated_print_time_s=estimated_time,
            estimated_volume_mm3=estimated_volume,
        )

        logger.info(
            f"Multi-strategy plan: {num_layers} layers, "
            f"{strategy_count}, {total_length:.0f} mm total, "
            f"~{estimated_time:.0f}s print time"
        )
        return plan

    def _plan_honeycomb_layer(
        self, layer_idx: int, depth_mm: float, fill_amount: float,
        boundary_mm: NDArray[np.float64],
    ) -> LayerPlan:
        """Plan a honeycomb scaffold layer."""
        from multistrategy.toolpath.honeycomb_planner import plan_honeycomb

        result = plan_honeycomb(
            boundary_mm, depth_mm,
            hex_size=self.config.hex_cell_size_mm,
            nozzle_dia=self.config.nozzle_diameter_mm,
        )
        return LayerPlan(
            layer_index=layer_idx,
            strategy=LayerStrategy.HONEYCOMB,
            fill_amount=fill_amount,
            depth_mm=depth_mm,
            waypoints_mm=result["waypoints"],
            normals=result["normals"],
            is_deposition=result["is_deposition"],
            path_length_mm=result["path_length_mm"],
            num_cells=result["num_cells"],
        )

    def _plan_geodesic_layer(
        self, layer_idx: int, depth_mm: float, fill_amount: float,
        boundary_mm: NDArray[np.float64], surface_mesh=None,
    ) -> LayerPlan:
        """Plan a geodesic conformal layer."""
        from multistrategy.toolpath.geodesic_planner import plan_geodesic

        result = plan_geodesic(
            boundary_mm, depth_mm,
            spacing_mm=self.config.geodesic_spacing_mm,
            adaptive_curvature=self.config.adaptive_curvature,
            curvature_factor=self.config.curvature_factor,
            surface_mesh=surface_mesh,
        )
        return LayerPlan(
            layer_index=layer_idx,
            strategy=LayerStrategy.GEODESIC,
            fill_amount=fill_amount,
            depth_mm=depth_mm,
            waypoints_mm=result["waypoints"],
            normals=result["normals"],
            is_deposition=result["is_deposition"],
            path_length_mm=result["path_length_mm"],
        )

    def _plan_hybrid_layer(
        self, layer_idx: int, depth_mm: float, fill_amount: float,
        boundary_mm: NDArray[np.float64], surface_mesh=None,
    ) -> LayerPlan:
        """Plan a hybrid layer: honeycomb core + geodesic perimeter."""
        from multistrategy.toolpath.hybrid_planner import plan_hybrid

        result = plan_hybrid(
            boundary_mm, depth_mm,
            hex_size=self.config.hex_cell_size_mm,
            geodesic_spacing=self.config.geodesic_spacing_mm,
            perimeter_fraction=self.config.hybrid_perimeter_fraction,
            surface_mesh=surface_mesh,
        )
        return LayerPlan(
            layer_index=layer_idx,
            strategy=LayerStrategy.HYBRID,
            fill_amount=fill_amount,
            depth_mm=depth_mm,
            waypoints_mm=result["waypoints"],
            normals=result["normals"],
            is_deposition=result["is_deposition"],
            path_length_mm=result["path_length_mm"],
            num_cells=result.get("num_cells", 0),
        )
