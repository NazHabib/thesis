import pandas as pd
from asgiref.sync import sync_to_async
from fastapi import APIRouter, HTTPException

from apps.api.schemas import ColdStartRunRequest, ColdStartRunResponse

router = APIRouter(prefix="/cold-start", tags=["cold-start"])


@sync_to_async
def run_cold_start_service(payload: ColdStartRunRequest) -> dict:
    from apps.recs_csl.services.cold_start.service import run_cold_start_layer

    cfm_df = pd.DataFrame(payload.cfm)
    participants_df = pd.DataFrame(payload.participants)

    result = run_cold_start_layer(cfm_df, participants_df)

    return {
        "safety_context": result["safety_context"].to_dict(orient="records"),
        "gated_candidates": result["gated_candidates"].to_dict(orient="records"),
    }


@router.post("/run", response_model=ColdStartRunResponse)
async def run_cold_start(payload: ColdStartRunRequest):
    try:
        return await run_cold_start_service(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
