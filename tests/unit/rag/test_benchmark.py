"""The benchmark's arithmetic and, more importantly, its two stop conditions.

No model. The encoder calls arrive as injected callables precisely so that the
projection, the thresholds and the wording are testable, because those are the
parts that decide whether a multi-hour run should start.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from signaldesk.rag.index.benchmark import (
    MAX_HEALTHY_RATIO,
    MIN_HEALTHY_RATIO,
    BenchmarkReport,
    TokenReport,
    measure,
    render,
)

pytestmark = pytest.mark.unit

TEXTS = ["lactic acidosis has been reported", "hepatic failure", "rash", "nausea and vomiting"]
PAIRS = [("Boxed warning", text) for text in TEXTS]


def _measure(
    *,
    lengths: list[int] | None = None,
    width: int = 768,
    pending: int = 1_000,
    **overrides: object,
) -> BenchmarkReport:
    real = lengths if lengths is not None else [20, 12, 6, 14]
    return measure(
        TEXTS,
        PAIRS,
        token_lengths=lambda _inputs: real,
        encode=lambda inputs: np.ones((len(inputs), width)),
        score=lambda _query, texts: np.ones(len(texts)),
        pending_chunks=pending,
        rerank_candidates=overrides.get("rerank_candidates", 100),  # type: ignore[arg-type]
        load_seconds=1.5,
        peak_rss_bytes=512 * 1024**2,
        limit=512,
        expected_dimensions=768,
    )


def _report(**overrides: object) -> TokenReport:
    fields: dict[str, object] = {
        "sampled": 10,
        "estimated_total": 1_300,
        "real_total": 1_000,
        "max_real": 200,
        "truncated": 0,
        "limit": 512,
    }
    fields.update(overrides)
    return TokenReport(**fields)  # type: ignore[arg-type]


class TestTheProjection:
    def test_throughput_is_the_sample_over_the_time_it_took(self) -> None:
        report = _measure()

        assert report.chunks_per_second > 0
        assert report.projected_seconds == pytest.approx(
            report.pending_chunks / report.chunks_per_second
        )

    def test_a_larger_backlog_projects_a_longer_run(self) -> None:
        assert _measure(pending=100_000).projected_seconds > _measure(pending=100).projected_seconds

    def test_nothing_measured_projects_nothing_rather_than_dividing_by_zero(self) -> None:
        report = BenchmarkReport(
            sample=0,
            load_seconds=0.0,
            measured_dimensions=768,
            expected_dimensions=768,
            tokens=_report(),
            encode_seconds=0.0,
            pending_chunks=500,
            rerank_candidates=0,
            rerank_seconds=0.0,
            peak_rss_bytes=0,
        )

        assert report.chunks_per_second == 0.0
        assert report.projected_seconds == 0.0
        assert "unknown" in render(report)

    def test_the_rerank_cost_is_reported_per_query(self) -> None:
        """The number that decides whether the reranker is usable interactively."""
        assert "one query" in render(_measure())


class TestTruncation:
    def test_no_truncation_is_not_a_blocking_condition(self) -> None:
        assert _measure(lengths=[20, 12, 6, 14]).blocked is False

    def test_any_truncation_blocks_the_run(self) -> None:
        """Text past the cut is text the index would claim to hold and would not."""
        assert _measure(lengths=[600, 12, 6, 14]).blocked is True

    def test_the_output_says_stop_rather_than_printing_a_number(self) -> None:
        rendered = render(_measure(lengths=[600, 700, 6, 14]))

        assert "STOP." in rendered
        assert "2 of 4 sampled chunks exceed 512 tokens" in rendered
        assert "claims to hold and does not" in rendered
        assert "before a multi-hour run" in rendered

    def test_a_clean_sample_says_a_run_can_start(self) -> None:
        assert "No blocking condition" in render(_measure())


class TestTheWidthCheck:
    def test_a_disagreeing_width_blocks_the_run(self) -> None:
        assert _measure(width=384).blocked is True

    def test_the_output_names_both_widths_and_says_stop(self) -> None:
        rendered = render(_measure(width=384))

        assert "STOP." in rendered
        assert "384-dimensional" in rendered
        assert "768" in rendered
        assert "Do not start a corpus run" in rendered


class TestTheEstimatorVerdict:
    def test_erring_high_by_the_intended_margin_is_healthy(self) -> None:
        report = _report(estimated_total=1_300, real_total=1_000)

        assert report.healthy is True
        assert "errs high by the intended margin" in report.verdict

    def test_under_counting_is_called_out_as_the_dangerous_direction(self) -> None:
        """Predicting fewer tokens than the tokenizer produces loses text."""
        report = _report(estimated_total=900, real_total=1_000)

        assert report.healthy is False
        assert "UNDER-counting" in report.verdict
        assert "TOKENS_PER_WORD" in report.verdict

    def test_over_counting_heavily_is_called_out_as_costly_but_lossless(self) -> None:
        report = _report(estimated_total=2_500, real_total=1_000)

        assert report.healthy is False
        assert "over-counting heavily" in report.verdict
        assert "loses no text" in report.verdict

    @pytest.mark.parametrize("ratio", [MIN_HEALTHY_RATIO, MAX_HEALTHY_RATIO])
    def test_the_thresholds_are_inclusive(self, ratio: float) -> None:
        assert _report(estimated_total=int(1_000 * ratio), real_total=1_000).healthy is True

    def test_the_verdict_appears_in_the_rendered_output_in_words(self) -> None:
        """Addition B: a ratio the reader has to interpret is not a report."""
        rendered = render(_measure())

        assert "estimator:" in rendered

    def test_a_drifting_estimator_does_not_by_itself_block_the_run(self) -> None:
        """It is a tuning signal. Only lost text and a wrong width are stops."""
        report = _measure(lengths=[1, 1, 1, 1])

        assert report.tokens.healthy is False
        assert report.blocked is False


class TestTheSample:
    def test_the_encoder_is_timed_on_the_real_input_shape(self) -> None:
        """Pairs, not texts: timing single texts would time a shape we never send."""
        seen: list[Sequence[object]] = []

        measure(
            TEXTS,
            PAIRS,
            token_lengths=lambda inputs: [10] * len(inputs),
            encode=lambda inputs: (seen.append(inputs), np.ones((len(inputs), 768)))[1],
            score=lambda _q, texts: np.ones(len(texts)),
            pending_chunks=10,
            rerank_candidates=2,
            load_seconds=0.0,
            peak_rss_bytes=0,
            limit=512,
            expected_dimensions=768,
        )

        assert seen[0] == PAIRS

    def test_the_rerank_sample_is_capped_at_the_candidate_pool(self) -> None:
        assert _measure(rerank_candidates=2).rerank_candidates == 2
