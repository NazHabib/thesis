from fastapi import APIRouter, HTTPException

from apps.api.schemas import RecommendationRequest

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/run")
async def run_recommendations(payload: RecommendationRequest):
    raise HTTPException(
        status_code=501,
        detail=(
            "This endpoint is intentionally disabled in the thesis build. "
            "Patient recommendation generation is handled through the Django "
            "patient portal using the live single-patient pipeline."
        ),
    )
