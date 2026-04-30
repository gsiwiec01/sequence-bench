from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class EpochMetric(Base):
    __tablename__ = "epoch_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    train_loss: Mapped[float | None] = mapped_column(Float)
    val_loss: Mapped[float | None] = mapped_column(Float)
    metric_value: Mapped[float | None] = mapped_column(Float)
    epoch_time_s: Mapped[float | None] = mapped_column(Float)
    gpu_memory_mb: Mapped[float | None] = mapped_column(Float)

    experiment: Mapped["Experiment"] = relationship(back_populates="epoch_metrics")


class GradientLog(Base):
    __tablename__ = "gradient_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)

    experiment: Mapped["Experiment"] = relationship(back_populates="gradient_logs")
