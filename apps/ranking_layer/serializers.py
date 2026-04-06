from rest_framework import serializers


class RankingRunSerializer(serializers.Serializer):
    safety_context = serializers.JSONField()
    gated_candidates = serializers.JSONField()
