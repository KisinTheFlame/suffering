"""Artifact storage helpers for baseline training outputs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from suffering.config.settings import Settings, get_settings


class TrainingStorage:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.models_dir = self.artifacts_dir / "models"
        self.reports_dir = self.artifacts_dir / "reports"
        self.predictions_dir = self.artifacts_dir / "predictions"

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "TrainingStorage":
        resolved_settings = settings or get_settings()
        return cls(artifacts_dir=resolved_settings.artifacts_dir)

    def model_path(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.pkl"

    def metrics_report_path(self, model_name: str) -> Path:
        return self.reports_dir / f"{model_name}_metrics.json"

    def prediction_path(self, model_name: str, split_name: str) -> Path:
        return self.predictions_dir / f"{model_name}_{split_name}.csv"

    def walkforward_summary_path(self, model_name: str) -> Path:
        return self.reports_dir / f"{model_name}_walkforward_summary.json"

    def walkforward_folds_path(self, model_name: str) -> Path:
        return self.reports_dir / f"{model_name}_walkforward_folds.csv"

    def walkforward_predictions_path(self, model_name: str) -> Path:
        return self.predictions_dir / f"{model_name}_walkforward_test_predictions.csv"

    def write_model(self, model_name: str, model: Any) -> Path:
        path = self.model_path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(model, file)
        return path

    def write_metrics_report(self, model_name: str, report: dict[str, Any]) -> Path:
        path = self.metrics_report_path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        return path

    def read_metrics_report(self, model_name: str) -> dict[str, Any]:
        path = self.metrics_report_path(model_name)
        if not path.exists():
            raise FileNotFoundError(f"Training metrics report not found for model: {model_name}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def write_walkforward_summary(self, model_name: str, report: dict[str, Any]) -> Path:
        path = self.walkforward_summary_path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        return path

    def read_walkforward_summary(self, model_name: str) -> dict[str, Any]:
        path = self.walkforward_summary_path(model_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Walk-forward summary report not found for model: {model_name}"
            )
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def write_predictions(
        self,
        model_name: str,
        split_name: str,
        frame: pd.DataFrame,
    ) -> Path:
        path = self.prediction_path(model_name, split_name)
        return self._write_frame(path, frame)

    def write_walkforward_folds(self, model_name: str, frame: pd.DataFrame) -> Path:
        path = self.walkforward_folds_path(model_name)
        return self._write_frame(path, frame)

    def write_walkforward_predictions(self, model_name: str, frame: pd.DataFrame) -> Path:
        path = self.walkforward_predictions_path(model_name)
        return self._write_frame(path, frame)

    def _write_frame(self, path: Path, frame: pd.DataFrame) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        output = frame.copy()
        if "date" in output.columns:
            output["date"] = pd.to_datetime(output["date"]).dt.tz_localize(None)
        output.to_csv(path, index=False, date_format="%Y-%m-%d")
        return path
