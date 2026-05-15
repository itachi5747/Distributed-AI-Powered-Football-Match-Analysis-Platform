from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine
from app.routers import auth, matches

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("FootballIQ API starting", env=settings.APP_ENV)
    yield
    log.info("FootballIQ API shutting down")
    await engine.dispose()


app = FastAPI(
    title="FootballIQ API",
    description="AI-powered football match analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "env": settings.APP_ENV}


app.include_router(auth.router)
app.include_router(matches.router)
