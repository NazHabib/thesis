import uuid
from django.db import models


class CollaborativeSnapshot(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=64, unique=True)
    is_active = models.BooleanField(default=False)

    similarity_method = models.CharField(
        max_length=16,
        choices=[("cosine", "cosine"), ("euclidean", "euclidean")],
        default="cosine",
    )
    k_neighbors = models.PositiveSmallIntegerField(default=20)
    similarity_threshold = models.FloatField(default=0.60)
    min_neighbors = models.PositiveSmallIntegerField(default=5)
    feature_schema_version = models.CharField(max_length=32, default="v1")

    source_ranked_path = models.CharField(max_length=512)
    source_profile_path = models.CharField(max_length=512)
    preprocessor_path = models.CharField(max_length=512)
    matrix_path = models.CharField(max_length=512)
    index_path = models.CharField(max_length=512)

    manifest = models.JSONField(default=dict)
    built_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["is_active", "built_at"], name="collab_snap_active_idx"),
        ]


class ReferenceRankedAction(models.Model):
    snapshot = models.ForeignKey(
        CollaborativeSnapshot,
        on_delete=models.CASCADE,
        related_name="reference_actions",
    )
    participant_key = models.BigIntegerField()
    action_code = models.CharField(max_length=128)
    bundle_family = models.CharField(max_length=64)
    score_norm = models.FloatField()
    rank_pre_ml = models.PositiveSmallIntegerField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["snapshot", "participant_key", "action_code"],
                name="uq_ref_action_snapshot_pt_action",
            )
        ]
        indexes = [
            models.Index(fields=["snapshot", "participant_key"], name="ref_action_snap_pt_idx"),
            models.Index(fields=["snapshot", "action_code"], name="ref_action_snap_action_idx"),
        ]


class ParticipantNeighbor(models.Model):
    snapshot = models.ForeignKey(
        CollaborativeSnapshot,
        on_delete=models.CASCADE,
        related_name="neighbors",
    )
    participant_key = models.BigIntegerField()
    neighbor_participant_key = models.BigIntegerField()
    neighbor_rank = models.PositiveSmallIntegerField()
    similarity = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["snapshot", "participant_key", "neighbor_participant_key"],
                name="uq_neighbor_snapshot_pt_neighbor",
            )
        ]
        indexes = [
            models.Index(
                fields=["snapshot", "participant_key", "neighbor_rank"],
                name="neighbor_snapshot_pt_rank_idx",
            )
        ]


class CollaborativeRecommendationFeature(models.Model):
    snapshot = models.ForeignKey(
        CollaborativeSnapshot,
        on_delete=models.CASCADE,
        related_name="features",
    )
    participant_key = models.BigIntegerField()
    action_code = models.CharField(max_length=128)
    bundle_family = models.CharField(max_length=64)

    collab_available_flag = models.BooleanField(default=False)
    collab_neighbor_count = models.PositiveSmallIntegerField(default=0)
    collab_support_count = models.PositiveSmallIntegerField(default=0)
    collab_support_rate = models.FloatField(default=0.0)
    collab_mean_rule_score_norm = models.FloatField(default=0.0)
    collab_weighted_rule_score_norm = models.FloatField(default=0.0)
    collab_mean_rank_pre_ml = models.FloatField(default=0.0)
    collab_family_support_count = models.PositiveSmallIntegerField(default=0)
    collab_family_support_rate = models.FloatField(default=0.0)
    collab_similarity_mean = models.FloatField(default=0.0)
    collab_similarity_max = models.FloatField(default=0.0)
    collab_method = models.CharField(max_length=16, default="cosine")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["snapshot", "participant_key", "action_code"],
                name="uq_collab_feat_snapshot_pt_action",
            )
        ]
        indexes = [
            models.Index(fields=["snapshot", "participant_key"], name="collab_feat_snap_pt_idx"),
            models.Index(
                fields=["snapshot", "participant_key", "bundle_family"],
                name="collab_feat_snap_pt_family_idx",
            ),
        ]
