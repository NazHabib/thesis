from asgiref.sync import sync_to_async
from fastapi import APIRouter, HTTPException

from apps.api.schemas import ExplanationRequest

router = APIRouter(prefix="/explainability", tags=["explainability"])


@sync_to_async
def build_explanation_service(participant_id: str, bundle_code: str) -> dict:
    return {
        "participant_id": participant_id,
        "bundle_code": bundle_code,
        "explanation": "placeholder",
        "status": "ok",
    }


@router.post("/bundle")
async def bundle_explanation(payload: ExplanationRequest):
    try:
        return await build_explanation_service(
            participant_id=payload.participant_id,
            bundle_code=payload.bundle_code,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
