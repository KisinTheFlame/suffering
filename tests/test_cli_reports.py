from pathlib import Path

from suffering.cli import main


class FakeReportService:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def generate_research_report(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, object]:
        resolved_model_name = model_name or "xgb_ranker"
        report_path = self.root_dir / f"{resolved_model_name}_research_report.md"
        report_path.write_text("# 研究报告：xgb_ranker\n", encoding="utf-8")
        return {
            "model_name": resolved_model_name,
            "task_type": "ranking",
            "available_artifacts": [{"name": "walkforward_summary", "path": "a.json"}],
            "missing_artifacts": [{"name": "robustness_summary", "path": "b.json"}],
            "available_sections": ["walkforward", "backtest", "benchmark_comparison"],
            "missing_sections": ["robustness"],
            "report_path": str(report_path),
        }

    def read_research_report(self, model_name: str | None = None) -> dict[str, str]:
        resolved_model_name = model_name or "xgb_ranker"
        if resolved_model_name == "missing":
            raise FileNotFoundError
        report_path = self.root_dir / f"{resolved_model_name}_research_report.md"
        report_path.write_text(
            "# 研究报告：xgb_ranker\n\n## 执行摘要\n\n- 示例\n",
            encoding="utf-8",
        )
        return {
            "model_name": resolved_model_name,
            "report_path": str(report_path),
            "content": report_path.read_text(encoding="utf-8"),
        }


def test_report_generate_command_can_be_called(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_report_service",
        lambda settings=None: FakeReportService(tmp_path),
    )

    exit_code = main(["report-generate", "--model", "xgb_ranker"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model: xgb_ranker" in captured.out
    assert "available_artifacts: walkforward_summary" in captured.out
    assert "missing_sections: robustness" in captured.out


def test_report_show_command_can_be_called(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_report_service",
        lambda settings=None: FakeReportService(tmp_path),
    )

    exit_code = main(["report-show", "--model", "xgb_ranker", "--full"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "report_path:" in captured.out
    assert "# 研究报告：xgb_ranker" in captured.out


def test_report_show_command_reports_missing_file(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_report_service",
        lambda settings=None: FakeReportService(tmp_path),
    )

    exit_code = main(["report-show", "--model", "missing"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "research report not found for model=missing" in captured.out
