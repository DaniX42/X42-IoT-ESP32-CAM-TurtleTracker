from datetime import datetime

from pydantic import BaseModel, Field


class Position(BaseModel):
    timestamp: datetime
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    inside_house: bool = False
    speed: float = Field(default=0, ge=0)
    confidence: float = Field(default=0, ge=0, le=1)


class IngestResponse(BaseModel):
    accepted: bool
    position: Position | None = None
    reason: str | None = None


class HeatmapPoint(BaseModel):
    x: float
    y: float
    count: int
