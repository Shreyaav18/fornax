import logging
import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select
from api.schemas import GenerateRequest, GenerateResponse
from db.database import get_async_session
from db.models import GeneratedOutput, TrainingRun
from inference.generate import generate
from tokenizer.tokenizer_utils import load_tokenizer
from model.transformer import GPT
from config.model_config import ModelConfig
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generate"])

TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "tokenizer_data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache: dict = {}

def load_model_for_run(run: TrainingRun) -> GPT:
    if run.id in _model_cache:
        logger.info(f"Using cached model for run_id={run.id}")
        return _model_cache[run.id]

    if not run.checkpoint_path or not os.path.exists(run.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for run_id={run.id} at {run.checkpoint_path}")

    try:
        model_config = ModelConfig(**run.model_config)
        model = GPT(model_config).to(DEVICE)

        checkpoint = torch.load(run.checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _model_cache[run.id] = model
        logger.info(f"Model loaded for run_id={run.id} from {run.checkpoint_path}")
        return model

    except Exception as e:
        logger.error(f"Model load failed for run_id={run.id}: {e}")
        raise


@router.post("/", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    session: AsyncSession = Depends(get_async_session)
):
    try:
        result = await session.execute(
            select(TrainingRun).where(TrainingRun.id == request.run_id)
        )
        run = result.scalar_one_or_none()

        if run is None:
            raise HTTPException(status_code=404, detail=f"Training run {request.run_id} not found")

        if run.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Run {request.run_id} has status '{run.status}' — only completed runs can generate"
            )

    except SQLAlchemyError as e:
        logger.error(f"Database error fetching run {request.run_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching training run")

    try:
        model = load_model_for_run(run)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model for generation")

    try:
        tokenizer = load_tokenizer(TOKENIZER_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Tokenizer loading failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to load tokenizer")

    try:
        output_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            greedy=request.greedy,
            repetition_penalty=request.repetition_penalty,
            device=DEVICE
        )
    except Exception as e:
        logger.error(f"Text generation failed for run_id={request.run_id}: {e}")
        raise HTTPException(status_code=500, detail="Text generation failed")

    try:
        record = GeneratedOutput(
            run_id=request.run_id,
            prompt=request.prompt,
            output=output_text,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            greedy=int(request.greedy),
            repetition_penalty=request.repetition_penalty
        )
        session.add(record)
        await session.flush()
        await session.refresh(record)

        logger.info(f"Generated output saved — id={record.id}, run_id={record.run_id}")

    except SQLAlchemyError as e:
        logger.error(f"Failed to save generated output: {e}")
        raise HTTPException(status_code=500, detail="Failed to save generated output")

    return GenerateResponse(
        id=record.id,
        run_id=record.run_id,
        prompt=record.prompt,
        output=record.output,
        max_new_tokens=record.max_new_tokens,
        temperature=record.temperature,
        top_k=record.top_k,
        top_p=record.top_p,
        greedy=bool(record.greedy),
        repetition_penalty=record.repetition_penalty,
        generated_at=record.generated_at
    )