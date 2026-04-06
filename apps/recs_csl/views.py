import pandas as pd
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from .serializers import ColdStartRunSerializer
from .services.cold_start.service import run_cold_start_layer


class ColdStartRunView(APIView):
    def post(self, request):
        serializer = ColdStartRunSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cfm_df = pd.DataFrame(serializer.validated_data['cfm'])
        participants_df = pd.DataFrame(serializer.validated_data['participants'])

        result = run_cold_start_layer(cfm_df, participants_df)

        return Response(
            {
                'safety_context': result['safety_context'].to_dict(orient='records'),
                'gated_candidates': result['gated_candidates'].to_dict(orient='records'),
            },
            status=status.HTTP_200_OK,
        )
