import pandas as pd
from asgiref.sync import sync_to_async
from fastapi import APIRouter, HTTPException

from apps.api.schemas import PipelineRunRequest, PipelineRunResponse

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@sync_to_async
def run_pipeline_service(payload: PipelineRunRequest) -> dict:
    from apps.recs_csl.services.cold_start.service import run_cold_start_layer
    from apps.ranking_layer.services.ranking.service import run_ranking_layer
    from apps.ml_layer.services.ml_reranking.service import run_ml_reranking_layer
    from apps.ml_layer.services.ml_reranking.adapter import prepare_reranked_for_bundling
    from apps.bundling_layer.services.bundling.service import run_bundling_layer
    from apps.explainability_layer.services.explainability.service import run_explainability_layer

    cfm_df = pd.DataFrame(payload.cfm)
    participants_df = pd.DataFrame(payload.participants)

    cold_start_result = run_cold_start_layer(
        cfm_df=cfm_df,
        participants_df=participants_df,
    )

    safety_context_df = cold_start_result["safety_context"]
    gated_candidates_df = cold_start_result["gated_candidates"]

    ranking_result = run_ranking_layer(
        safety_context_df=safety_context_df,
        gated_candidates_df=gated_candidates_df,
    )

    ranked_df = ranking_result["ranked_recommendations"]

    ml_result = run_ml_reranking_layer(
        safety_context_df=safety_context_df,
        ranked_df=ranked_df,
        participant_features_df=cfm_df,
        fusion_strategy=payload.fusion_strategy,
    )

    reranked_df = ml_result["reranked_recommendations"]
    bundling_input_df = prepare_reranked_for_bundling(reranked_df)

    bundling_result = run_bundling_layer(
        safety_context_df=safety_context_df,
        ranked_df=bundling_input_df,
    )

    final_recommendations_df = bundling_result["final_recommendations"]

    explainability_result = run_explainability_layer(
        safety_context_df=safety_context_df,
        final_recommendations_df=final_recommendations_df,
        reranked_df=reranked_df,
        participant_features_df=cfm_df,
    )

    explained_df = explainability_result["explained_recommendations"]

    return {
        "status": "success",
        "safety_context": safety_context_df.to_dict(orient="records"),
        "gated_candidates": gated_candidates_df.to_dict(orient="records"),
        "ranked_recommendations": ranked_df.to_dict(orient="records"),
        "reranked_recommendations": reranked_df.to_dict(orient="records"),
        "final_recommendations": final_recommendations_df.to_dict(orient="records"),
        "explained_recommendations": explained_df.to_dict(orient="records"),
        "cold_start_manifest": cold_start_result.get("manifest", {}),
        "ranking_manifest": ranking_result.get("manifest", {}),
        "ml_reranking_manifest": ml_result.get("manifest", {}),
        "bundling_manifest": bundling_result.get("manifest", {}),
        "explainability_manifest": explainability_result.get("manifest", {}),
    }


@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(payload: PipelineRunRequest):
    try:
        return await run_pipeline_service(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
