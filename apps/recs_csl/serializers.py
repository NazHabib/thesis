from rest_framework import serializers


class ColdStartRunSerializer(serializers.Serializer):
    cfm = serializers.JSONField()
    participants = serializers.JSONField()
