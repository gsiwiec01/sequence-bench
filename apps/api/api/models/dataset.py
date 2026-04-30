import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum as SAEnum, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class DatasetType(str, enum.Enum):
    BUILTIN = "builtin"
    CUSTOM = "custom"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True,
                                    default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[DatasetType] = mapped_column(SAEnum(DatasetType), nullable=False)
    task_type: Mapped[str] = mapped_column(String, nullable=False)
    T: Mapped[int] = mapped_column(Integer, nullable=False)
    input_size: Mapped[int] = mapped_column(Integer, nullable=False)
    output_size: Mapped[int] = mapped_column(Integer, nullable=False)
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    experiments: Mapped[list["Experiment"]] = relationship(back_populates="dataset")
