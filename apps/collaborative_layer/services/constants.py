COLLAB_VERSION = "collab_1.0.0"

K_NEIGHBORS_DEFAULT = 20
SIMILARITY_THRESHOLD_DEFAULT = 0.45
MIN_NEIGHBORS_DEFAULT = 5

COLLAB_DEFAULTS = {
    "collab_available_flag": 0,
    "collab_neighbor_count": 0,
    "collab_support_count": 0,
    "collab_support_rate": 0.0,
    "collab_mean_rule_score_norm": 0.0,
    "collab_mean_rank_pre_ml": 0.0,
    "collab_family_support_rate": 0.0,

    # weighted collaborative features
    "collab_weighted_support_rate": 0.0,
    "collab_weighted_mean_rule_score_norm": 0.0,
    "collab_weighted_mean_rank_pre_ml": 0.0,
    "collab_mean_supporting_similarity": 0.0,

    # safety/explanation
    "collab_safety_interpretation": "",
}

COLLAB_FEATURE_COLUMNS = [
    "ParticipantKey",
    "action_code",
    "bundle_family",
    "collab_available_flag",
    "collab_neighbor_count",
    "collab_support_count",
    "collab_support_rate",
    "collab_mean_rule_score_norm",
    "collab_mean_rank_pre_ml",
    "collab_family_support_rate",
    "collab_weighted_support_rate",
    "collab_weighted_mean_rule_score_norm",
    "collab_weighted_mean_rank_pre_ml",
    "collab_mean_supporting_similarity",
    "collab_safety_interpretation",
]
