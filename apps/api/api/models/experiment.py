import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum as SAEnum, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

if TYPE_CHECKING:
    from .dataset import Dataset
    from .metrics import EpochMetric, GradientLog


class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True,
                                    default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    architecture: Mapped[str] = mapped_column(String, nullable=False)
    k1: Mapped[int] = mapped_column(Integer, nullable=False)
    k2: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparams: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    status: Mapped[ExperimentStatus] = mapped_column(
        SAEnum(ExperimentStatus), default=ExperimentStatus.PENDING
    )
    best_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="experiments")
    epoch_metrics: Mapped[list["EpochMetric"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    gradient_logs: Mapped[list["GradientLog"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
