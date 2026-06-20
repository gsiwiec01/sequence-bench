import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.models.database import Base

if TYPE_CHECKING:
    from api.models.experiment import Experiment

class EpochMetric(Base):
    __tablename__ = "epoch_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    train_loss: Mapped[float | None] = mapped_column(Float)
    val_loss: Mapped[float | None] = mapped_column(Float)
    epoch_time_s: Mapped[float | None] = mapped_column(Float)
    gpu_memory_mb: Mapped[float | None] = mapped_column(Float)
    grad_norm_mean: Mapped[float | None] = mapped_column(Float)
    grad_norm_max: Mapped[float | None] = mapped_column(Float)
    learning_rate: Mapped[float | None] = mapped_column(Float)

    experiment: Mapped["Experiment"] = relationship(back_populates="epoch_metrics")


class GradientLog(Base):
    __tablename__ = "gradient_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    grad_norm_mean: Mapped[float | None] = mapped_column(Float)
    n_layers: Mapped[int | None] = mapped_column(Integer)
    n_steps: Mapped[int | None] = mapped_column(Integer)

    experiment: Mapped["Experiment"] = relationship(back_populates="gradient_logs")


class LossLandscape(Base):
    __tablename__ = "loss_landscapes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    cache_key: Mapped[str | None] = mapped_column(String, nullable=True, unique=True, index=True)
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    x_range: Mapped[list | None] = mapped_column(JSON, nullable=True)
    y_range: Mapped[list | None] = mapped_column(JSON, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    explained_variance: Mapped[float | None] = mapped_column(Float, nullable=True)
    anchor_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="queued")
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    experiment: Mapped["Experiment"] = relationship(back_populates="loss_landscapes")


class AdditionalMetric(Base):
    __tablename__ = "additional_metrics"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)

    experiment: Mapped["Experiment"] = relationship(back_populates="additional_metric_rows")

    __table_args__ = (
        Index(
            "ix_additional_metrics_exp_epoch_name",
            "experiment_id", "epoch", "metric_name",
        ),
    )
