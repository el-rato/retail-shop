"""FastAPI application entry point.

Creates the app, wires middleware (CORS, logging, exception handlers),
registers routers, initialises the database and the unified ML pipeline,
and exposes /health plus auto-generated Swagger docs at /docs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger
from app.database.session import init_db
from app.routers import chatbot, classify, dashboard, recognize, sentiment
from app.services.pipeline import get_pipeline

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    logger.info("Starting %s v%s (%s)", settings.APP_NAME, settings.APP_VERSION, settings.ENV)
    init_db()
    # Warm the ML pipeline eagerly so the first request is fast.
    try:
        get_pipeline()
    except Exception as exc:  # pragma: no cover - depends on optional deps
        logger.error("ML pipeline failed to warm up: %s", exc)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI-Powered Smart Retail & Customer Intelligence Platform.\n\n"
            "Endpoints: face recognition, product classification, sentiment "
            "analysis, FAQ chatbot, and dashboard statistics. Authenticate "
            "with the `X-API-Key` header."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(recognize.router)
    app.include_router(classify.router)
    app.include_router(sentiment.router)
    app.include_router(chatbot.router)
    app.include_router(dashboard.router)

    @app.get("/health", tags=["Health"], summary="Health check")
    def health() -> dict:
        return {
            "status": "ok",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENV,
        }

    return app


app = create_app()
