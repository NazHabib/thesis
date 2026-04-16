from asgiref.sync import sync_to_async
from fastapi import APIRouter, HTTPException

from apps.api.schemas import RecommendationRequest

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@sync_to_async
def run_recommendation_service(participant_id: str) -> dict:
    """
    Import Django-dependent or heavy service code inside the function,
    not at module import time.
    """
    # Example:
    # from apps.recs_csl.services.some_service import run_pipeline
    # result = run_pipeline(participant_id)

    result = {
        "participant_id": participant_id,
        "recommendations": [],
        "status": "ok",
    }
    return result


@router.post("/run")
async def run_recommendations(payload: RecommendationRequest):
    try:
        result = await run_recommendation_service(payload.participant_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
