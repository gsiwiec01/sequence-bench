import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.models.database import Base

if TYPE_CHECKING:
    from api.models.experiment import Experiment

class ExperimentGroup(Base):
    __tablename__ = "experiment_groups"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    dataset: Mapped[str | None] = mapped_column(String, nullable=True)
    created_from_matrix: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    experiments: Mapped[list["Experiment"]] = relationship(
        back_populates="group",
        foreign_keys="Experiment.group_id",
    )
