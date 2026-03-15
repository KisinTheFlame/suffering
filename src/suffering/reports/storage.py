"""Storage helpers for markdown research reports."""

from __future__ import annotations

from pathlib import Path

from suffering.config.settings import Settings, get_settings


class ReportStorage:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.research_reports_dir = self.artifacts_dir / "reports" / "research"

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ReportStorage":
        resolved_settings = settings or get_settings()
        return cls(artifacts_dir=resolved_settings.artifacts_dir)

    def research_report_path(self, model_name: str) -> Path:
        return self.research_reports_dir / f"{model_name}_research_report.md"

    def write_research_report(self, model_name: str, markdown_text: str) -> Path:
        path = self.research_report_path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_text, encoding="utf-8")
        return path

    def read_research_report(self, model_name: str) -> str:
        path = self.research_report_path(model_name)
        if not path.exists():
            raise FileNotFoundError(f"Research report not found for model: {model_name}")
        return path.read_text(encoding="utf-8")
