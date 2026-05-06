from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from apps.api.routers import (
    health,
    cold_start,
    ranking,
    bundling,
    pipeline,
    explainability,
)

fastapi_app = FastAPI(
    title="Safety-First Hybrid Clinical Recommendation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    default_response_class=ORJSONResponse,
)

fastapi_app.include_router(health.router)
fastapi_app.include_router(cold_start.router)
fastapi_app.include_router(ranking.router)
fastapi_app.include_router(bundling.router)
fastapi_app.include_router(pipeline.router)
fastapi_app.include_router(explainability.router)
