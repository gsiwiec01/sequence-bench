import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum as SAEnum, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.models.database import Base

if TYPE_CHECKING:
    from api.models.dataset import Dataset
    from api.models.group import ExperimentGroup
    from api.models.metrics import AdditionalMetric, EpochMetric, GradientLog, LossLandscape

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
    task_type: Mapped[str] = mapped_column(String, nullable=False, server_default="classification")
    early_stopping_metric: Mapped[str] = mapped_column(String, nullable=False, server_default="val_loss")
    early_stopping_mode: Mapped[str] = mapped_column(String, nullable=False, server_default="min")
    hyperparams: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    status: Mapped[ExperimentStatus] = mapped_column(
        SAEnum(ExperimentStatus), default=ExperimentStatus.PENDING
    )
    best_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    n_parameters: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_training_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    full_weight_path: Mapped[str | None] = mapped_column(String, nullable=True)
    convergence_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)

    group_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiment_groups.id", ondelete="SET NULL"), nullable=True, index=True
    )

    dataset: Mapped["Dataset"] = relationship(back_populates="experiments")
    group: Mapped["ExperimentGroup | None"] = relationship(back_populates="experiments")
    epoch_metrics: Mapped[list["EpochMetric"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    gradient_logs: Mapped[list["GradientLog"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    additional_metric_rows: Mapped[list["AdditionalMetric"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    loss_landscapes: Mapped[list["LossLandscape"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )

    @property
    def has_weight_trajectory(self) -> bool:
        return self.full_weight_path is not None

    @property
    def additional_metrics(self) -> list[str]:
        from api.metrics_registry import AVAILABLE_METRICS
        return AVAILABLE_METRICS.get(self.task_type, [])
