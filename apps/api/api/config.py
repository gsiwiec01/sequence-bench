from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_connection_string: str = "postgresql+asyncpg://user:pass@localhost:5432/sequence_bench"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Storage
    gradient_storage_dir: str = "./storage/gradients"
    upload_storage_dir: str = "./storage/uploads"
    checkpoint_dir: str = "./storage/checkpoints"
    max_upload_size_mb: int = 500

    # API
    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
