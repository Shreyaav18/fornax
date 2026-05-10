import logging
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select
from db.models import TrainingRun

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def create_training_run(
    session: AsyncSession,
    model_config: dict,
    train_config: dict,
    name: str = "fornax"
) -> TrainingRun:
    try:
        run = TrainingRun(
            name=name,
            status="running",
            model_config=model_config,
            train_config=train_config,
            started_at=datetime.utcnow()
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        logger.info(f"TrainingRun created — id={run.id}, name={run.name}")
        return run

    except SQLAlchemyError as e:
        logger.error(f"Failed to create training run: {e}")
        await session.rollback()
        raise


async def update_checkpoint(
    session: AsyncSession,
    run_id: int,
    checkpoint_path: str,
    best_val_loss: float,
    step: int
) -> Optional[TrainingRun]:
    try:
        result = await session.execute(
            select(TrainingRun).where(TrainingRun.id == run_id)
        )
        run = result.scalar_one_or_none()

        if run is None:
            logger.warning(f"TrainingRun id={run_id} not found for checkpoint update")
            return None

        run.checkpoint_path = checkpoint_path
        run.best_val_loss = best_val_loss
        run.final_step = step

        await session.commit()
        await session.refresh(run)
        logger.info(f"TrainingRun id={run_id} checkpoint updated — step={step}, val_loss={best_val_loss:.4f}")
        return run

    except SQLAlchemyError as e:
        logger.error(f"Failed to update checkpoint for run_id={run_id}: {e}")
        await session.rollback()
        raise


async def complete_training_run(
    session: AsyncSession,
    run_id: int,
    final_step: int,
    best_val_loss: float
) -> Optional[TrainingRun]:
    try:
        result = await session.execute(
            select(TrainingRun).where(TrainingRun.id == run_id)
        )
        run = result.scalar_one_or_none()

        if run is None:
            logger.warning(f"TrainingRun id={run_id} not found for completion update")
            return None

        run.status = "completed"
        run.final_step = final_step
        run.best_val_loss = best_val_loss
        run.finished_at = datetime.utcnow()

        await session.commit()
        await session.refresh(run)
        logger.info(f"TrainingRun id={run_id} marked as completed — final_step={final_step}, best_val_loss={best_val_loss:.4f}")
        return run

    except SQLAlchemyError as e:
        logger.error(f"Failed to complete training run id={run_id}: {e}")
        await session.rollback()
        raise


async def fail_training_run(
    session: AsyncSession,
    run_id: int
) -> Optional[TrainingRun]:
    try:
        result = await session.execute(
            select(TrainingRun).where(TrainingRun.id == run_id)
        )
        run = result.scalar_one_or_none()

        if run is None:
            logger.warning(f"TrainingRun id={run_id} not found for failure update")
            return None

        run.status = "failed"
        run.finished_at = datetime.utcnow()

        await session.commit()
        await session.refresh(run)
        logger.info(f"TrainingRun id={run_id} marked as failed")
        return run

    except SQLAlchemyError as e:
        logger.error(f"Failed to mark training run id={run_id} as failed: {e}")
        await session.rollback()
        raise