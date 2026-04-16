import pandas as pd
from asgiref.sync import sync_to_async
from fastapi import APIRouter, HTTPException

from apps.api.schemas import PipelineRunRequest, PipelineRunResponse

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@sync_to_async
def run_pipeline_service(payload: PipelineRunRequest) -> dict:
    from apps.recs_csl.services.cold_start.service import run_cold_start_layer
    from apps.ranking_layer.services.ranking.service import run_ranking_layer
    from apps.bundling_layer.services.bundling.service import run_bundling_layer

    cfm_df = pd.DataFrame(payload.cfm)
    participants_df = pd.DataFrame(payload.participants)

    cold_start_result = run_cold_start_layer(cfm_df, participants_df)

    safety_context_df = cold_start_result["safety_context"]
    gated_candidates_df = cold_start_result["gated_candidates"]

    ranking_result = run_ranking_layer(safety_context_df, gated_candidates_df)

    ranked_df = ranking_result["ranked_recommendations"]

    bundling_result = run_bundling_layer(safety_context_df, ranked_df)

    return {
        "status": "success",
        "safety_context": safety_context_df.to_dict(orient="records"),
        "gated_candidates": gated_candidates_df.to_dict(orient="records"),
        "ranked_recommendations": ranked_df.to_dict(orient="records"),
        "final_recommendations": bundling_result["final_recommendations"].to_dict(orient="records"),
        "ranking_manifest": ranking_result["manifest"],
        "bundling_manifest": bundling_result["manifest"],
    }


@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(payload: PipelineRunRequest):
    try:
        return await run_pipeline_service(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
