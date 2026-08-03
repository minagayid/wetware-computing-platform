from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import platform
from typing import Any


@dataclass(frozen=True)
class RunManifest:
    schema: str
    model_schema: str
    seed: int
    dt_ms: float
    configuration: dict
    input_digest: str
    event_ordering: str = "timestamp_then_event_id"
    runtime: str = ""

    @classmethod
    def create(cls, reservoir, input_digest: str) -> "RunManifest":
        return cls(
            schema="wetware-platform.manifest.v1",
            model_schema=reservoir.SNAPSHOT_SCHEMA,
            seed=reservoir.config.seed,
            dt_ms=reservoir.config.dt_ms,
            configuration=asdict(reservoir.config),
            input_digest=str(input_digest),
            runtime=platform.python_version(),
        )

    def canonical_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
