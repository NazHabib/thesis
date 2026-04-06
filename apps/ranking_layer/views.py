import pandas as pd
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import RankingRunSerializer
from .services.ranking.service import run_ranking_layer


class RankingRunView(APIView):
    def post(self, request):
        serializer = RankingRunSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            safety_context_df = pd.DataFrame(serializer.validated_data["safety_context"])
            gated_candidates_df = pd.DataFrame(serializer.validated_data["gated_candidates"])

            result = run_ranking_layer(safety_context_df, gated_candidates_df)

            return Response(
                {
                    "status": "success",
                    "ranked_recommendations": result["ranked_recommendations"].to_dict(orient="records"),
                    "manifest": result["manifest"],
                },
                status=status.HTTP_200_OK,
            )
        except Exception as exc:
            return Response(
                {
                    "status": "error",
                    "message": str(exc),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
