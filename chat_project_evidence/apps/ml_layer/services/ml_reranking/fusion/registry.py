from __future__ import annotations

from .strategies import MinMaxAdditiveFusion, WeightedAverageFusion


DEFAULT_FUSION_STRATEGY = "minmax_additive_v1"


FUSION_REGISTRY = {
    MinMaxAdditiveFusion.name: MinMaxAdditiveFusion(),
    WeightedAverageFusion.name: WeightedAverageFusion(),
}


def resolve_fusion_strategy(name: str | None = None):
    strategy_name = name or DEFAULT_FUSION_STRATEGY

    if strategy_name not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion strategy: {strategy_name}. "
            f"Available: {sorted(FUSION_REGISTRY)}"
        )

    return FUSION_REGISTRY[strategy_name]
