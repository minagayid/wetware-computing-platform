"""Read-only local recording loader with strict validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterator, Tuple


@dataclass(frozen=True)
class RecordedFrame:
    timestamp_ms: float
    values: Tuple[float, ...]


class RecordedDataSource:
    """Immutable, read-only JSON recording source; never emits device commands."""

    def __init__(self, frames: Tuple[RecordedFrame, ...], channel_count: int):
        self.frames = frames
        self.channel_count = channel_count

    @classmethod
    def from_json(cls, path: str | Path) -> "RecordedDataSource":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "channel_count", "frames"}:
            raise ValueError("recording requires schema_version, channel_count, and frames")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported recording schema version")
        channel_count = int(payload["channel_count"])
        if channel_count <= 0:
            raise ValueError("channel_count must be positive")
        frames = []
        previous = -float("inf")
        for index, raw in enumerate(payload["frames"]):
            if not isinstance(raw, dict) or set(raw) != {"timestamp_ms", "values"}:
                raise ValueError(f"frame {index} has unknown or missing fields")
            timestamp = float(raw["timestamp_ms"])
            values = tuple(float(value) for value in raw["values"])
            if not math.isfinite(timestamp) or timestamp < 0.0 or timestamp < previous:
                raise ValueError(f"frame {index} has an invalid or out-of-order timestamp")
            if len(values) != channel_count or not all(math.isfinite(value) for value in values):
                raise ValueError(f"frame {index} has invalid channel values")
            frames.append(RecordedFrame(timestamp, values))
            previous = timestamp
        return cls(tuple(frames), channel_count)

    def __iter__(self) -> Iterator[Tuple[float, ...]]:
        for frame in self.frames:
            yield frame.values
