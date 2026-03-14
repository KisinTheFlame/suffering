"""High-level baseline training service."""

from __future__ import annotations

from typing import Any

import pandas as pd

from suffering.config.settings import Settings, get_settings
from suffering.data.models import DATE_COLUMN
from suffering.ranking import RankingService, build_ranking_service
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.training.baseline import train_hist_gradient_boosting_baseline
from suffering.training.evaluate import evaluate_predictions
from suffering.training.splits import split_panel_dataset_by_date
from suffering.training.storage import TrainingStorage


class TrainingService:
    def __init__(
        self,
        storage: TrainingStorage,
        ranking_service: RankingService,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.ranking_service = ranking_service
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "TrainingService":
        resolved_settings = settings or get_settings()
        return cls(
            storage=TrainingStorage.from_settings(resolved_settings),
            ranking_service=build_ranking_service(settings=resolved_settings),
            settings=resolved_settings,
        )

    def train_baseline(
        self,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        resolved_dataset_name = dataset_name or self.settings.default_dataset_name
        resolved_model_name = model_name or self.settings.default_model_name
        dataset_frame = self.ranking_service.read_panel_dataset(dataset_name=resolved_dataset_name)

        split = split_panel_dataset_by_date(
            frame=dataset_frame,
            train_ratio=self.settings.default_train_ratio,
            validation_ratio=self.settings.default_validation_ratio,
            test_ratio=self.settings.default_test_ratio,
        )
        training_result = train_hist_gradient_boosting_baseline(
            train_frame=split.train_frame,
            validation_frame=split.validation_frame,
            test_frame=split.test_frame,
            random_state=self.settings.random_seed,
        )

        validation_metrics = evaluate_predictions(training_result.validation_predictions)
        test_metrics = evaluate_predictions(training_result.test_predictions)
        split_summary = {
            "train": _build_split_summary(split.train_frame),
            "validation": _build_split_summary(split.validation_frame),
            "test": _build_split_summary(split.test_frame),
        }

        model_path = self.storage.write_model(resolved_model_name, training_result.model)
        metrics_path = self.storage.write_metrics_report(
            resolved_model_name,
            {
                "dataset_name": resolved_dataset_name,
                "model_name": resolved_model_name,
                "target_column": FUTURE_RETURN_5D_COLUMN,
                "feature_columns": training_result.feature_columns,
                "feature_count": len(training_result.feature_columns),
                "total_rows": int(len(dataset_frame)),
                "split_summary": split_summary,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            },
        )
        validation_predictions_path = self.storage.write_predictions(
            resolved_model_name,
            "validation",
            training_result.validation_predictions,
        )
        test_predictions_path = self.storage.write_predictions(
            resolved_model_name,
            "test",
            training_result.test_predictions,
        )

        return {
            "dataset_name": resolved_dataset_name,
            "model_name": resolved_model_name,
            "total_rows": int(len(dataset_frame)),
            "feature_columns": training_result.feature_columns,
            "feature_count": len(training_result.feature_columns),
            "split_summary": split_summary,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "artifacts": {
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "validation_predictions_path": str(validation_predictions_path),
                "test_predictions_path": str(test_predictions_path),
            },
        }

    def read_training_report(self, model_name: str | None = None) -> dict[str, Any]:
        resolved_model_name = model_name or self.settings.default_model_name
        report = self.storage.read_metrics_report(resolved_model_name)
        return {
            **report,
            "artifacts": {
                "model_path": str(self.storage.model_path(resolved_model_name)),
                "metrics_path": str(self.storage.metrics_report_path(resolved_model_name)),
                "validation_predictions_path": str(
                    self.storage.prediction_path(resolved_model_name, "validation")
                ),
                "test_predictions_path": str(
                    self.storage.prediction_path(resolved_model_name, "test")
                ),
            },
        }


def _build_split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "date_count": 0, "date_start": None, "date_end": None}

    dates = pd.to_datetime(frame[DATE_COLUMN]).dt.tz_localize(None).dt.normalize()
    return {
        "rows": int(len(frame)),
        "date_count": int(dates.nunique()),
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
    }


def build_training_service(settings: Settings | None = None) -> TrainingService:
    return TrainingService.from_settings(settings=settings)
