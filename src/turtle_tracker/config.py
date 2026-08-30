from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_path: Path = Path("data/turtle_tracker.db")
    motion_crops_path: Path = Path("data/motion_crops")
    mock_images_enabled: bool = True
    enclosure_length_meters: float = 7.0
    enclosure_width_meters: float = 2.5
    min_confidence: float = 0.1
    mqtt_enabled: bool = False
    mqtt_host: str | None = None
    mqtt_port: int = 1883
    mqtt_user: str | None = None
    mqtt_password: str | None = None
    mqtt_topic_prefix: str = "turtle_tracker/house"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
