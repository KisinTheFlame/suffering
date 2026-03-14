import pandas as pd

from suffering.config.settings import Settings
from suffering.training.evaluate import evaluate_predictions
from suffering.training.ranking import (
    SCORE_PREDICTION_COLUMN,
    build_date_query_groups,
    train_ranker,
)
from suffering.training.splits import split_panel_dataset_by_date


def build_training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, date_value in enumerate(pd.date_range("2024-01-02", periods=12, freq="B")):
        for symbol_index, symbol in enumerate(("AAPL", "MSFT", "NVDA", "META")):
            feature_alpha = float(date_index + symbol_index)
            feature_beta = float((date_index + 1) * (symbol_index + 1))
            rows.append(
                {
                    "date": date_value,
                    "symbol": symbol,
                    "feature_alpha": feature_alpha,
                    "feature_beta": feature_beta,
                    "future_return_5d": feature_alpha * 0.01 - feature_beta * 0.001,
                    "relevance_5d_5q": symbol_index,
                }
            )
    return pd.DataFrame(rows)


def test_build_date_query_groups_uses_one_group_per_date_with_stable_order() -> None:
    frame = build_training_frame()
    split = split_panel_dataset_by_date(
        frame=frame,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
    )

    train_groups = build_date_query_groups(split.train_frame)
    validation_groups = build_date_query_groups(split.validation_frame)
    test_groups = build_date_query_groups(split.test_frame)

    assert train_groups.group_sizes == [4, 4, 4, 4, 4, 4]
    assert validation_groups.group_sizes == [4, 4, 4]
    assert test_groups.group_sizes == [4, 4, 4]
    assert train_groups.qid.tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
    ]
    assert validation_groups.qid.min() == 0
    assert validation_groups.qid.max() == 2
    assert test_groups.qid.min() == 0
    assert test_groups.qid.max() == 2
    assert split.train_frame["date"].max() < split.validation_frame["date"].min()
    assert split.validation_frame["date"].max() < split.test_frame["date"].min()


def test_train_ranker_fits_predicts_and_returns_ndcg_metrics() -> None:
    frame = build_training_frame()
    split = split_panel_dataset_by_date(
        frame=frame,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    result = train_ranker(
        train_frame=split.train_frame,
        validation_frame=split.validation_frame,
        test_frame=split.test_frame,
        settings=Settings(xgb_ranker_n_estimators=12),
        random_state=7,
    )

    validation_metrics = evaluate_predictions(
        result.validation_predictions,
        prediction_column=SCORE_PREDICTION_COLUMN,
        include_error_metrics=False,
    )
    test_metrics = evaluate_predictions(
        result.test_predictions,
        prediction_column=SCORE_PREDICTION_COLUMN,
        include_error_metrics=False,
    )

    assert result.feature_columns == ["feature_alpha", "feature_beta"]
    assert len(result.validation_predictions) == len(split.validation_frame)
    assert len(result.test_predictions) == len(split.test_frame)
    assert SCORE_PREDICTION_COLUMN in result.validation_predictions.columns
    assert result.test_predictions["model_name"].eq("xgb_ranker").all()
    assert validation_metrics["mae"] is None
    assert test_metrics["overall_spearman_corr"] is not None
    assert test_metrics["ndcg_at_5_mean"] is not None
