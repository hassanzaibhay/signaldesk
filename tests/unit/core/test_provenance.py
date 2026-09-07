"""The two facts every committed artifact carries about its own run.

Extracted from ``analytics.signals`` once a second thing started writing
artifacts. Two copies of a provenance helper are two things that can drift while
both look authoritative, which is a worse failure than the duplication is a cost.
"""

from __future__ import annotations

import pytest

from signaldesk.core.provenance import code_sha, peak_rss_bytes

pytestmark = pytest.mark.unit


class TestCodeSha:
    def test_the_environment_wins_when_it_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """.git is not mounted into the container, so the Makefile passes it in."""
        monkeypatch.setenv("SIGNALDESK_CODE_SHA", "a" * 40)

        assert code_sha() == "a" * 40

    def test_surrounding_whitespace_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIGNALDESK_CODE_SHA", "  b1c2d3  ")

        assert code_sha() == "b1c2d3"

    def test_a_blank_environment_value_is_not_a_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls through to git rather than recording an empty commit."""
        monkeypatch.setenv("SIGNALDESK_CODE_SHA", "   ")

        assert code_sha() != ""

    def test_it_never_returns_an_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty string in an artifact looks like a value. 'unknown' says the
        provenance is incomplete, which is a different and worse thing."""
        monkeypatch.delenv("SIGNALDESK_CODE_SHA", raising=False)

        assert code_sha().strip()

    def test_a_git_that_cannot_run_reports_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIGNALDESK_CODE_SHA", raising=False)

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise OSError("no git here")

        monkeypatch.setattr("subprocess.run", _raise)

        assert code_sha() == "unknown"

    def test_a_failing_git_reports_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIGNALDESK_CODE_SHA", raising=False)

        class Failed:
            returncode = 128
            stdout = ""

        monkeypatch.setattr("subprocess.run", lambda *_a, **_k: Failed())

        assert code_sha() == "unknown"


class TestPeakRss:
    def test_it_reports_a_positive_number_of_bytes(self) -> None:
        assert peak_rss_bytes() > 0

    def test_it_is_reported_in_bytes_not_kilobytes(self) -> None:
        """ru_maxrss is kilobytes on Linux; a raw value would understate by 1024."""
        assert peak_rss_bytes() > 1024 * 1024
