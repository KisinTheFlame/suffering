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
from suffering.training.splits import build_frame_date_summary, split_panel_dataset_by_date
from suffering.training.storage import TrainingStorage
from suffering.training.walkforward import run_walkforward_baseline


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
            "train": build_frame_date_summary(split.train_frame),
            "validation": build_frame_date_summary(split.validation_frame),
            "test": build_frame_date_summary(split.test_frame),
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

    def train_walkforward(
        self,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        resolved_dataset_name = dataset_name or self.settings.default_dataset_name
        resolved_model_name = model_name or self.settings.default_model_name
        dataset_frame = self.ranking_service.read_panel_dataset(dataset_name=resolved_dataset_name)

        training_result = run_walkforward_baseline(
            frame=dataset_frame,
            validation_ratio=self.settings.default_validation_ratio,
            test_ratio=self.settings.default_test_ratio,
            step_ratio=self.settings.walkforward_step_ratio,
            random_state=self.settings.random_seed,
            min_folds=self.settings.walkforward_min_folds,
        )

        fold_records = _build_walkforward_fold_records(training_result.fold_results)
        notes = _build_walkforward_notes(len(training_result.fold_results))
        summary_report = {
            "dataset_name": resolved_dataset_name,
            "model_name": resolved_model_name,
            "target_column": FUTURE_RETURN_5D_COLUMN,
            "feature_columns": training_result.feature_columns,
            "feature_count": len(training_result.feature_columns),
            "total_rows": int(len(dataset_frame)),
            "date_count": int(
                pd.to_datetime(dataset_frame[DATE_COLUMN])
                .dt.tz_localize(None)
                .dt.normalize()
                .nunique()
            ),
            "fold_count": len(training_result.fold_results),
            "validation_ratio": self.settings.default_validation_ratio,
            "test_ratio": self.settings.default_test_ratio,
            "step_ratio": self.settings.walkforward_step_ratio,
            "test_metrics_summary": training_result.test_metrics_summary,
            "notes": notes,
        }

        summary_path = self.storage.write_walkforward_summary(resolved_model_name, summary_report)
        folds_path = self.storage.write_walkforward_folds(
            resolved_model_name,
            pd.DataFrame(fold_records),
        )
        predictions_path = self.storage.write_walkforward_predictions(
            resolved_model_name,
            training_result.combined_test_predictions,
        )

        return {
            **summary_report,
            "folds": fold_records,
            "artifacts": {
                "summary_path": str(summary_path),
                "folds_path": str(folds_path),
                "predictions_path": str(predictions_path),
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

    def read_walkforward_report(self, model_name: str | None = None) -> dict[str, Any]:
        resolved_model_name = model_name or self.settings.default_model_name
        report = self.storage.read_walkforward_summary(resolved_model_name)
        return {
            **report,
            "artifacts": {
                "summary_path": str(self.storage.walkforward_summary_path(resolved_model_name)),
                "folds_path": str(self.storage.walkforward_folds_path(resolved_model_name)),
                "predictions_path": str(
                    self.storage.walkforward_predictions_path(resolved_model_name)
                ),
            },
        }


def build_training_service(settings: Settings | None = None) -> TrainingService:
    return TrainingService.from_settings(settings=settings)


def _build_walkforward_fold_records(
    fold_results: list[Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in fold_results:
        train_summary = result.split_summary["train"]
        validation_summary = result.split_summary["validation"]
        test_summary = result.split_summary["test"]
        records.append(
            {
                "fold_id": result.fold_id,
                "train_rows": train_summary["rows"],
                "train_date_count": train_summary["date_count"],
                "train_date_start": train_summary["date_start"],
                "train_date_end": train_summary["date_end"],
                "validation_rows": validation_summary["rows"],
                "validation_date_count": validation_summary["date_count"],
                "validation_date_start": validation_summary["date_start"],
                "validation_date_end": validation_summary["date_end"],
                "test_rows": test_summary["rows"],
                "test_date_count": test_summary["date_count"],
                "test_date_start": test_summary["date_start"],
                "test_date_end": test_summary["date_end"],
                **result.test_metrics,
            }
        )
    return records


def _build_walkforward_notes(fold_count: int) -> list[str]:
    if fold_count < 2:
        return ["Only one walk-forward fold was generated because unique dates are limited."]
    return []
