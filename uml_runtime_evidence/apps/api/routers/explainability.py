from fastapi import APIRouter, HTTPException

from apps.api.schemas import ExplanationRequest

router = APIRouter(prefix="/explainability", tags=["explainability"])


@router.post("/bundle")
async def bundle_explanation(payload: ExplanationRequest):
    raise HTTPException(
        status_code=501,
        detail=(
            "This endpoint is intentionally disabled in the thesis build. "
            "Explainability is generated as part of the full pipeline and shown "
            "through the Django patient, clinician, and thesis evidence views."
        ),
    )
