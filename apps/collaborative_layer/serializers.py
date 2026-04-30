# apps/collaborative_layer/serializers.py
from rest_framework import serializers


class CandidateActionSerializer(serializers.Serializer):
    action_code = serializers.CharField(max_length=128)
    bundle_family = serializers.CharField(max_length=64)


class CollaborativeResolveRequestSerializer(serializers.Serializer):
    participant_key = serializers.IntegerField()
    snapshot_id = serializers.UUIDField(required=False)
    use_online_if_missing = serializers.BooleanField(default=True)
    candidates = CandidateActionSerializer(many=True)


class CollaborativeFeatureSerializer(serializers.Serializer):
    ParticipantKey = serializers.IntegerField()
    action_code = serializers.CharField()
    bundle_family = serializers.CharField()
    collab_available_flag = serializers.BooleanField()
    collab_neighbor_count = serializers.IntegerField()
    collab_support_count = serializers.IntegerField()
    collab_support_rate = serializers.FloatField()
    collab_mean_rule_score_norm = serializers.FloatField()
    collab_weighted_rule_score_norm = serializers.FloatField()
    collab_mean_rank_pre_ml = serializers.FloatField()
    collab_family_support_rate = serializers.FloatField()
