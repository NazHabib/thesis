from rest_framework import serializers


class BundlingRunSerializer(serializers.Serializer):
    safety_context = serializers.JSONField()
    ranked_recommendations = serializers.JSONField()
