import pandas as pd

from suffering.training.splits import split_panel_dataset_by_date


def build_panel_frame(
    date_count: int = 5,
    symbols: tuple[str, ...] = ("AAPL", "MSFT"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, date_value in enumerate(
        pd.date_range("2024-01-02", periods=date_count, freq="B")
    ):
        for symbol_index, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": date_value,
                    "symbol": symbol,
                    "feature_alpha": float(date_index + symbol_index),
                    "future_return_5d": float(date_index - symbol_index) / 100,
                    "relevance_5d_5q": symbol_index,
                }
            )
    return pd.DataFrame(rows)


def test_split_panel_dataset_by_date_keeps_dates_disjoint_and_ordered() -> None:
    frame = build_panel_frame()

    split = split_panel_dataset_by_date(
        frame=frame,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    train_dates = set(split.train_frame["date"].dt.strftime("%Y-%m-%d"))
    validation_dates = set(split.validation_frame["date"].dt.strftime("%Y-%m-%d"))
    test_dates = set(split.test_frame["date"].dt.strftime("%Y-%m-%d"))

    assert len(train_dates) == 3
    assert len(validation_dates) == 1
    assert len(test_dates) == 1
    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)
    assert split.train_frame["date"].max() < split.validation_frame["date"].min()
    assert split.validation_frame["date"].max() < split.test_frame["date"].min()


def test_split_panel_dataset_by_date_requires_at_least_three_dates() -> None:
    frame = build_panel_frame(date_count=2)

    try:
        split_panel_dataset_by_date(
            frame=frame,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
        )
    except ValueError as exc:
        assert "At least 3 unique dates" in str(exc)
    else:
        raise AssertionError("expected too-few-dates split to raise ValueError")
