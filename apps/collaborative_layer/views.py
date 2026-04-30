# apps/collaborative_layer/views.py
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.collaborative_layer.serializers import (
    CollaborativeResolveRequestSerializer,
    CollaborativeFeatureSerializer,
)
from apps.collaborative_layer.services.resolver import CollaborativeResolver


class CollaborativeResolveView(APIView):
    authentication_classes = [SessionAuthentication, TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        req = CollaborativeResolveRequestSerializer(data=request.data)
        req.is_valid(raise_exception=True)

        resolver = CollaborativeResolver()
        rows = resolver.resolve_features(
            participant_key=req.validated_data["participant_key"],
            candidates=req.validated_data["candidates"],
            snapshot_id=req.validated_data.get("snapshot_id"),
            use_online_if_missing=req.validated_data["use_online_if_missing"],
        )

        out = CollaborativeFeatureSerializer(rows, many=True)
        return Response({"rows": out.data})
