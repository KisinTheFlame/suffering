import pandas as pd

from suffering.ranking.panel import RELEVANCE_5D_5Q_COLUMN, build_daily_panel_dataset


def build_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "symbol": ["GOOGL", "MSFT", "META", "AAPL", "AAPL", "GOOGL", "AMZN", "MSFT"],
            "feature_alpha": [7.0, 2.0, 5.0, 6.0, 1.0, 3.0, 4.0, 8.0],
            "feature_beta": [70.0, 20.0, 50.0, 60.0, 10.0, 30.0, 40.0, 80.0],
        }
    )


def build_label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "symbol": ["GOOGL", "MSFT", "META", "AAPL", "AAPL", "GOOGL", "AMZN", "MSFT"],
            "future_return_5d": [None, -0.01, 0.10, 0.01, -0.05, 0.02, 0.04, 0.03],
        }
    )


def test_build_daily_panel_dataset_merges_sorts_and_generates_relevance() -> None:
    panel_frame = build_daily_panel_dataset(
        feature_frames=[build_feature_frame()],
        label_frames=[build_label_frame()],
    )

    assert list(panel_frame.columns) == [
        "date",
        "symbol",
        "feature_alpha",
        "feature_beta",
        "future_return_5d",
        RELEVANCE_5D_5Q_COLUMN,
    ]
    assert len(panel_frame) == 7
    assert panel_frame[["date", "symbol"]].values.tolist() == [
        [pd.Timestamp("2024-01-02"), "AAPL"],
        [pd.Timestamp("2024-01-02"), "AMZN"],
        [pd.Timestamp("2024-01-02"), "GOOGL"],
        [pd.Timestamp("2024-01-02"), "META"],
        [pd.Timestamp("2024-01-02"), "MSFT"],
        [pd.Timestamp("2024-01-03"), "AAPL"],
        [pd.Timestamp("2024-01-03"), "MSFT"],
    ]
    assert panel_frame["future_return_5d"].notna().all()

    relevance_map = {
        (row.date, row.symbol): row.relevance_5d_5q for row in panel_frame.itertuples(index=False)
    }
    assert relevance_map[(pd.Timestamp("2024-01-02"), "AAPL")] == 0
    assert relevance_map[(pd.Timestamp("2024-01-02"), "MSFT")] == 1
    assert relevance_map[(pd.Timestamp("2024-01-02"), "GOOGL")] == 2
    assert relevance_map[(pd.Timestamp("2024-01-02"), "AMZN")] == 3
    assert relevance_map[(pd.Timestamp("2024-01-02"), "META")] == 4
    assert relevance_map[(pd.Timestamp("2024-01-03"), "AAPL")] == 0
    assert relevance_map[(pd.Timestamp("2024-01-03"), "MSFT")] == 1
