"""Read-only local recording loader with strict validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterator, Tuple


MAX_RECORDING_BYTES = 16 * 1024 * 1024
MAX_RECORDING_FRAMES = 100_000
MAX_RECORDING_SAMPLES = 1_000_000


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
        with Path(path).open("rb") as handle:
            raw = handle.read(MAX_RECORDING_BYTES + 1)
        if len(raw) > MAX_RECORDING_BYTES:
            raise ValueError("recording exceeds the file-size limit")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "channel_count", "frames"}:
            raise ValueError("recording requires schema_version, channel_count, and frames")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported recording schema version")
        channel_count = payload["channel_count"]
        if isinstance(channel_count, bool) or not isinstance(channel_count, int) or not 1 <= channel_count <= MAX_RECORDING_SAMPLES:
            raise ValueError("channel_count must be a positive bounded integer")
        raw_frames = payload["frames"]
        if not isinstance(raw_frames, list):
            raise ValueError("recording frames must be an array")
        if len(raw_frames) > MAX_RECORDING_FRAMES or len(raw_frames) * channel_count > MAX_RECORDING_SAMPLES:
            raise ValueError("recording exceeds the frame or sample limit")
        frames = []
        previous = -float("inf")
        for index, raw in enumerate(raw_frames):
            if not isinstance(raw, dict) or set(raw) != {"timestamp_ms", "values"}:
                raise ValueError(f"frame {index} has unknown or missing fields")
            if not isinstance(raw["values"], list):
                raise ValueError(f"frame {index} values must be an array")
            try:
                timestamp = float(raw["timestamp_ms"])
                values = tuple(float(value) for value in raw["values"])
            except (OverflowError, TypeError, ValueError):
                raise ValueError(f"frame {index} contains non-numeric values") from None
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
