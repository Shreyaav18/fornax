import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select
from typing import List
from api.schemas import RunResponse
from db.database import get_async_session
from db.models import TrainingRun

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/", response_model=List[RunResponse])
async def list_runs(
    session: AsyncSession = Depends(get_async_session)
):
    try:
        result = await session.execute(
            select(TrainingRun).order_by(TrainingRun.started_at.desc())
        )
        runs = result.scalars().all()

        if not runs:
            logger.info("No training runs found")

        return [RunResponse(
            id=run.id,
            name=run.name,
            status=run.status,
            best_val_loss=run.best_val_loss,
            final_step=run.final_step,
            started_at=run.started_at,
            finished_at=run.finished_at,
            checkpoint_path=run.checkpoint_path,
            notes=run.notes,
            model_config=run.model_config,
            train_config=run.train_config
        ) for run in runs]

    except SQLAlchemyError as e:
        logger.error(f"Database error fetching runs: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching training runs")


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: int,
    session: AsyncSession = Depends(get_async_session)
):
    try:
        result = await session.execute(
            select(TrainingRun).where(TrainingRun.id == run_id)
        )
        run = result.scalar_one_or_none()

        if run is None:
            raise HTTPException(status_code=404, detail=f"Training run {run_id} not found")

        return RunResponse(
            id=run.id,
            name=run.name,
            status=run.status,
            best_val_loss=run.best_val_loss,
            final_step=run.final_step,
            started_at=run.started_at,
            finished_at=run.finished_at,
            checkpoint_path=run.checkpoint_path,
            notes=run.notes,
            model_config=run.model_config,
            train_config=run.train_config
        )

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error fetching run {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching training run")