import pandas as pd
from asgiref.sync import sync_to_async
from fastapi import APIRouter, HTTPException

from apps.api.schemas import BundlingRunRequest, BundlingRunResponse

router = APIRouter(prefix="/bundling", tags=["bundling"])


@sync_to_async
def run_bundling_service(payload: BundlingRunRequest) -> dict:
    from apps.bundling_layer.services.bundling.service import run_bundling_layer

    safety_context_df = pd.DataFrame(payload.safety_context)
    ranked_df = pd.DataFrame(payload.ranked_recommendations)

    result = run_bundling_layer(safety_context_df, ranked_df)

    return {
        "status": "success",
        "final_recommendations": result["final_recommendations"].to_dict(orient="records"),
        "manifest": result["manifest"],
    }


@router.post("/run", response_model=BundlingRunResponse)
async def run_bundling(payload: BundlingRunRequest):
    try:
        return await run_bundling_service(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
