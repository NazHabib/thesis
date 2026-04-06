import pandas as pd
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import BundlingRunSerializer
from .services.bundling.service import run_bundling_layer


class BundlingRunView(APIView):
    def post(self, request):
        serializer = BundlingRunSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            safety_context_df = pd.DataFrame(serializer.validated_data["safety_context"])
            ranked_df = pd.DataFrame(serializer.validated_data["ranked_recommendations"])

            result = run_bundling_layer(safety_context_df, ranked_df)

            return Response(
                {
                    "status": "success",
                    "final_recommendations": result["final_recommendations"].to_dict(orient="records"),
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
