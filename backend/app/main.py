# ISL Bridge FastAPI application.

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.websocket import register_websocket_routes
from app.config.settings import settings
from app.inference.service import get_inference_service, reset_inference_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_inference_service()
    yield
    reset_inference_service()


app = FastAPI(
    title="ISL Bridge API",
    description="AI-based Indian Sign Language real-time translator — local MVP.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
register_websocket_routes(app)


@app.get("/")
def root() -> dict:
    return {
        "service": "isl-bridge-api",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/translate",
        "phase": 7,
    }
