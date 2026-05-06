import pandas as pd
from asgiref.sync import sync_to_async
from fastapi import APIRouter, HTTPException

from apps.api.schemas import RankingRunRequest, RankingRunResponse

router = APIRouter(prefix="/ranking", tags=["ranking"])


@sync_to_async
def run_ranking_service(payload: RankingRunRequest) -> dict:
    from apps.ranking_layer.services.ranking.service import run_ranking_layer

    safety_context_df = pd.DataFrame(payload.safety_context)
    gated_candidates_df = pd.DataFrame(payload.gated_candidates)

    result = run_ranking_layer(safety_context_df, gated_candidates_df)

    return {
        "status": "success",
        "ranked_recommendations": result["ranked_recommendations"].to_dict(orient="records"),
        "manifest": result["manifest"],
    }


@router.post("/run", response_model=RankingRunResponse)
async def run_ranking(payload: RankingRunRequest):
    try:
        return await run_ranking_service(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
