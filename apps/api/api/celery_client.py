from celery import Celery
from api.config import settings

celery_app = Celery(
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
