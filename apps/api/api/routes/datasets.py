import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.models.database import get_db
from api.models.dataset import Dataset, DatasetType

router = APIRouter()

BUILTIN_DATASETS = [
    {"name": "copy_task_T100", "task_type": "seq2seq", "T": 121, "input_size": 10,
     "output_size": 9, "metric_name": "accuracy"},
    {"name": "adding_problem_T200", "task_type": "regression", "T": 200, "input_size": 2,
     "output_size": 1, "metric_name": "mse"},
    {"name": "sMNIST", "task_type": "classification", "T": 784, "input_size": 1,
     "output_size": 10, "metric_name": "accuracy"},
    {"name": "pMNIST", "task_type": "classification", "T": 784, "input_size": 1,
     "output_size": 10, "metric_name": "accuracy"},
    {"name": "penn_treebank", "task_type": "language_model", "T": 35, "input_size": 1,
     "output_size": 10000, "metric_name": "perplexity"},
    {"name": "ptbxl_ecg", "task_type": "classification", "T": 1000, "input_size": 12,
     "output_size": 5, "metric_name": "auc"},
    {"name": "speech_commands", "task_type": "classification", "T": 100, "input_size": 40,
     "output_size": 35, "metric_name": "accuracy"},
    {"name": "etth1_h96", "task_type": "regression", "T": 336, "input_size": 7,
     "output_size": 7, "metric_name": "mse"},
]


class DatasetResponse(BaseModel):
    id: str
    name: str
    type: str
    task_type: str
    T: int
    input_size: int
    output_size: int
    metric_name: str
    file_path: Optional[str]

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset))
    return result.scalars().all()


@router.post("/seed-builtins", status_code=201)
async def seed_builtin_datasets(db: AsyncSession = Depends(get_db)):
    created = []
    for meta in BUILTIN_DATASETS:
        existing = await db.execute(
            select(Dataset).where(
                Dataset.name == meta["name"],
                Dataset.type == DatasetType.BUILTIN,
            )
        )
        if existing.scalar_one_or_none():
            continue
        ds = Dataset(type=DatasetType.BUILTIN, **meta)
        db.add(ds)
        created.append(meta["name"])
    await db.commit()
    return {"created": created}


@router.post("/upload", response_model=DatasetResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    config: str = Form(...),
    name: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    file.file.seek(0, 2)
    size_mb = file.file.tell() / 1024 / 1024
    file.file.seek(0)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(413, f"Plik za duży: {size_mb:.1f} MB > {settings.max_upload_size_mb} MB")

    try:
        config_dict = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(422, "config musi być poprawnym JSON")

    from ml_engine.datasets.upload import UploadConfig, load_and_validate
    try:
        upload_config = UploadConfig(**config_dict)
    except Exception as e:
        raise HTTPException(422, f"Błędny config: {e}")

    upload_dir = Path(settings.upload_storage_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    suffix = Path(file.filename or "data.npz").suffix
    file_path = upload_dir / f"{file_id}{suffix}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        data = load_and_validate(str(file_path), upload_config)
    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(422, str(e))

    meta = data["metadata"]
    ds = Dataset(
        name=name,
        type=DatasetType.CUSTOM,
        task_type=upload_config.task_type,
        T=meta["T"],
        input_size=meta["d"],
        output_size=upload_config.n_classes or 1,
        metric_name="accuracy" if upload_config.task_type == "classification" else "mse",
        file_path=str(file_path),
        config_json=config_dict,
    )
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    return ds


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404)
    return ds


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404)
    if ds.type == DatasetType.BUILTIN:
        raise HTTPException(403, "Nie można usunąć wbudowanego datasetu")
    if ds.file_path:
        Path(ds.file_path).unlink(missing_ok=True)
    await db.delete(ds)
    await db.commit()


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    n: int = 5,
    db: AsyncSession = Depends(get_db),
):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404)
    if ds.type == DatasetType.BUILTIN or not ds.file_path:
        return {"message": "Podgląd niedostępny dla wbudowanych datasetów"}

    import numpy as np
    try:
        data = np.load(ds.file_path, allow_pickle=False)
        X = data["X"][:n].tolist()
        y = data["y"][:n].tolist() if "y" in data else None
    except Exception as e:
        raise HTTPException(500, f"Błąd odczytu pliku: {e}")

    return {"X": X, "y": y, "shape": list(data["X"].shape), "n_shown": min(n, len(X))}
