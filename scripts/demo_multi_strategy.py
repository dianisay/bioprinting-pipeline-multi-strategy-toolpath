"""Demo: Multi-strategy bioprinting pipeline.

Shows how the StrategyRouter decomposes a wound into multiple layers,
each with a different deposition strategy.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from multistrategy.toolpath.strategy_router import StrategyRouter, StrategyConfig


def main():
    print("=" * 70)
    print("MULTI-STRATEGY BIOPRINTING DEMO")
    print("=" * 70)

    # Simulate a wound: elliptical boundary, variable depth
    num_radii = 64
    num_layers = 4
    angles = np.linspace(0, 2 * np.pi, num_radii, endpoint=False)

    # Elliptical wound boundary (20mm x 15mm)
    boundary_mm = np.column_stack([
        20.0 * np.cos(angles),
        15.0 * np.sin(angles),
    ])

    # Depth varies: deeper in center, shallower at edges
    depth_profile_mm = 4.0 + 1.0 * np.sin(2 * angles)

    # Simulated decoder layer_amounts prediction:
    # Layer 0 (deepest): high fill (structural)
    # Layer 1: high fill (structural)
    # Layer 2: medium fill (transition)
    # Layer 3 (surface): low fill (conformal coverage)
    layer_fill_pattern = np.array([0.85, 0.70, 0.45, 0.18])
    layer_amounts = np.outer(np.ones(num_radii), layer_fill_pattern)

    print(f"\nWound: {boundary_mm.shape[0]} boundary points")
    print(f"  Size: ~{np.ptp(boundary_mm[:, 0]):.0f} x {np.ptp(boundary_mm[:, 1]):.0f} mm")
    print(f"  Depth: {depth_profile_mm.mean():.1f} mm (mean)")
    print(f"  Layers: {num_layers}")
    print(f"  Fill amounts: {layer_fill_pattern}")

    # Configure and run strategy router
    config = StrategyConfig(
        structural_threshold=0.6,
        conformal_threshold=0.3,
        hex_cell_size_mm=3.0,
        geodesic_spacing_mm=1.5,
    )
    router = StrategyRouter(config)

    print(f"\n{'─' * 70}")
    print("STRATEGY CLASSIFICATION")
    print(f"{'─' * 70}")

    strategies = router.classify_layers(layer_amounts, depth_profile_mm)
    for i, (strategy, fill) in enumerate(zip(strategies, layer_fill_pattern)):
        depth_at_layer = depth_profile_mm.max() - i * (depth_profile_mm.max() / num_layers)
        print(f"  Layer {i}: fill={fill:.2f} → {strategy.name:10s} (z={-depth_at_layer:.1f} mm)")

    print(f"\n{'─' * 70}")
    print("GENERATING MULTI-STRATEGY PLAN")
    print(f"{'─' * 70}")

    plan = router.plan(
        layer_amounts=layer_amounts,
        depth_profile_mm=depth_profile_mm,
        boundary_points_mm=boundary_mm,
    )

    print(f"\n{'═' * 70}")
    print("PLAN SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Total layers:      {plan.total_layers}")
    print(f"  Strategy breakdown: {plan.strategy_breakdown}")
    print(f"  Total path length: {plan.total_path_length_mm:.0f} mm")
    print(f"  Estimated time:    {plan.estimated_print_time_s:.0f} s ({plan.estimated_print_time_s/60:.1f} min)")
    print(f"  Estimated volume:  {plan.estimated_volume_mm3:.1f} mm³")

    print(f"\n  Per-layer detail:")
    for layer in plan.layers:
        dep_frac = layer.is_deposition.sum() / max(len(layer.is_deposition), 1) * 100
        cells_str = f", {layer.num_cells} cells" if layer.num_cells > 0 else ""
        print(f"    Layer {layer.layer_index}: {layer.strategy.name:10s} | "
              f"{layer.path_length_mm:6.0f} mm | "
              f"{dep_frac:.0f}% deposition{cells_str}")

    print(f"\n{'═' * 70}")
    print("DONE — Multi-strategy plan generated successfully")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
