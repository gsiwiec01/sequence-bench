from celery import Celery
from celery.signals import worker_init

from api.config import settings

celery_app = Celery(
    "sequence_bench",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["worker.training", "worker.landscape"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=False,
    task_reject_on_worker_lost=False,
    broker_transport_options={"visibility_timeout": 7200},
    result_expires=86400,
)

@worker_init.connect
def run_migrations_on_worker_start(**kwargs) -> None:
    from api.db_migrations import run_pending_migrations
    run_pending_migrations()
