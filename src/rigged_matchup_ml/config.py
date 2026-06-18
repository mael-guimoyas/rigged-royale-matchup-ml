from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    source_path: Path

    @property
    def database(self) -> dict[str, Any]:
        return self.raw["database"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    def resolve(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.source_path.parent.parent / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig(raw=raw, source_path=source)
