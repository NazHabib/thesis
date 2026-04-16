from typing import Any
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    participant_id: str


class ExplanationRequest(BaseModel):
    participant_id: str
    bundle_code: str


class ColdStartRunRequest(BaseModel):
    cfm: list[dict[str, Any]] = Field(default_factory=list)
    participants: list[dict[str, Any]] = Field(default_factory=list)


class ColdStartRunResponse(BaseModel):
    safety_context: list[dict[str, Any]]
    gated_candidates: list[dict[str, Any]]


class RankingRunRequest(BaseModel):
    safety_context: list[dict[str, Any]] = Field(default_factory=list)
    gated_candidates: list[dict[str, Any]] = Field(default_factory=list)


class RankingRunResponse(BaseModel):
    status: str
    ranked_recommendations: list[dict[str, Any]]
    manifest: dict[str, Any]


class BundlingRunRequest(BaseModel):
    safety_context: list[dict[str, Any]] = Field(default_factory=list)
    ranked_recommendations: list[dict[str, Any]] = Field(default_factory=list)


class BundlingRunResponse(BaseModel):
    status: str
    final_recommendations: list[dict[str, Any]]
    manifest: dict[str, Any]


class PipelineRunRequest(BaseModel):
    cfm: list[dict[str, Any]] = Field(default_factory=list)
    participants: list[dict[str, Any]] = Field(default_factory=list)


class PipelineRunResponse(BaseModel):
    status: str
    safety_context: list[dict[str, Any]]
    gated_candidates: list[dict[str, Any]]
    ranked_recommendations: list[dict[str, Any]]
    final_recommendations: list[dict[str, Any]]
    ranking_manifest: dict[str, Any]
    bundling_manifest: dict[str, Any]
