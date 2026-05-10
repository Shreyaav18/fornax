import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import generate, runs
from db.database import init_db, close_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_db()
        logger.info("Fornax API started — database initialized")
    except Exception as e:
        logger.error(f"Startup failed during database initialization: {e}")
        raise

    yield

    try:
        await close_db()
        logger.info("Fornax API shutdown — database connection closed")
    except Exception as e:
        logger.error(f"Shutdown error during database close: {e}")


app = FastAPI(
    title="Fornax API",
    description="Inference and training run management API for the Fornax transformer",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(generate.router)
app.include_router(runs.router)


@app.get("/health")
async def health():
    return {"status": "ok", "project": "fornax"}