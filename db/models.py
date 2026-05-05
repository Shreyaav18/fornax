import logging
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, relationship

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, default="fornax")
    status = Column(String(50), nullable=False, default="pending")
    model_config = Column(JSON, nullable=False)
    train_config = Column(JSON, nullable=False)
    best_val_loss = Column(Float, nullable=True)
    final_step = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    checkpoint_path = Column(String(512), nullable=True)
    notes = Column(Text, nullable=True)

    outputs = relationship("GeneratedOutput", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TrainingRun id={self.id} name={self.name} status={self.status}>"


class GeneratedOutput(Base):
    __tablename__ = "generated_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False)
    prompt = Column(Text, nullable=False)
    output = Column(Text, nullable=False)
    max_new_tokens = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=True)
    top_k = Column(Integer, nullable=True)
    top_p = Column(Float, nullable=True)
    greedy = Column(Integer, nullable=False, default=0)
    repetition_penalty = Column(Float, nullable=False, default=1.0)
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    run = relationship("TrainingRun", back_populates="outputs")

    def __repr__(self):
        return f"<GeneratedOutput id={self.id} run_id={self.run_id}>"