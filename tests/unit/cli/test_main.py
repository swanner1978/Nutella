"""CLI tests."""

from __future__ import annotations

from typer.testing import CliRunner

from nutella_scraper.cli.main import app


class TestCLI:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "nutella-scraper" in result.stdout
