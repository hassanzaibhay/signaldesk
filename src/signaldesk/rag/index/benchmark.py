"""Measuring what an embed run will cost, before committing hours to one.

The corpus embed is a multi-hour job on a machine with no accelerator, and every
figure anyone could quote about how long it takes is an estimate until something
times the real model on real chunks. This does that on a small sample and
projects, so the decision to start is made against a measurement.

It writes nothing. Throughput is a property of this machine, not of the corpus,
and ``evals/history/`` is for the second kind of fact.

Two of the things it reports are stop conditions rather than statistics:

* **Truncation.** The chunker sizes windows with a deliberately high estimate so
  that a chunk stays inside the encoder's 512 positions. If any sampled chunk
  would be truncated, that estimate is not doing its job, and the text past the
  cut is text the index would claim to hold and would not. That is a defect to
  fix before a long run, not a warning to read after one.
* **Estimator drift.** ``chunking.estimate_tokens`` drove the chunk sizing and has
  already had one defect where the character floor bound instead of the word
  bound. If the real wordpiece ratio has moved away from what the estimator
  assumes, the sizing rests on a number that is no longer true.

The model calls arrive as injected callables so that everything here - the
projection, the thresholds, the wording - is exercised by tests with no model.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from signaldesk.core.logging import get_logger
from signaldesk.rag.chunking import estimate_tokens
from signaldesk.stats.types import FloatArray

log = get_logger(__name__)

#: Below this, the estimator is under-counting and windows can exceed the
#: encoder's limit. This is the direction that loses text.
MIN_HEALTHY_RATIO: Final = 1.0

#: Above this, the estimator is so conservative that chunks come out far smaller
#: than the target, which costs embedding time and dilutes each vector.
MAX_HEALTHY_RATIO: Final = 1.8

#: A text pair or a text, whatever the encoder takes, to a list of real
#: wordpiece lengths with no truncation applied.
TokenLengths = Callable[[Sequence[object]], list[int]]

#: Encodes a batch and returns the vectors, for timing.
EncodeBatch = Callable[[Sequence[object]], FloatArray]

#: Scores one query against candidate texts, for timing.
ScoreBatch = Callable[[str, Sequence[str]], FloatArray]


@dataclass(frozen=True, slots=True)
class TokenReport:
    """How the chunker's estimate compares with the real tokenizer."""

    sampled: int
    estimated_total: int
    real_total: int
    max_real: int
    truncated: int
    limit: int

    @property
    def ratio(self) -> float:
        """Estimated tokens per real token. Above one means erring high."""
        return self.estimated_total / self.real_total if self.real_total else 0.0

    @property
    def healthy(self) -> bool:
        return MIN_HEALTHY_RATIO <= self.ratio <= MAX_HEALTHY_RATIO

    @property
    def verdict(self) -> str:
        """What the ratio means, in words rather than as a number to interpret."""
        if self.ratio < MIN_HEALTHY_RATIO:
            return (
                "the estimator is UNDER-counting: it predicts fewer tokens than the "
                "tokenizer produces, so windows can exceed the encoder limit and lose "
                "their tails. Raise TOKENS_PER_WORD in rag.chunking before embedding."
            )
        if self.ratio > MAX_HEALTHY_RATIO:
            return (
                "the estimator is over-counting heavily: chunks are coming out far "
                "smaller than the target, which costs embedding time and puts less "
                "context in each vector. Worth retuning, but it loses no text."
            )
        return "the estimator errs high by the intended margin; the sizing holds."


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """One machine, one sample, and what a corpus run would cost on it."""

    sample: int
    load_seconds: float
    measured_dimensions: int
    expected_dimensions: int
    tokens: TokenReport
    encode_seconds: float
    pending_chunks: int
    rerank_candidates: int
    rerank_seconds: float
    peak_rss_bytes: int

    @property
    def chunks_per_second(self) -> float:
        return self.sample / self.encode_seconds if self.encode_seconds > 0 else 0.0

    @property
    def projected_seconds(self) -> float:
        rate = self.chunks_per_second
        return self.pending_chunks / rate if rate > 0 else 0.0

    @property
    def blocked(self) -> bool:
        """Whether something here must be fixed before a corpus run."""
        return self.tokens.truncated > 0 or self.measured_dimensions != self.expected_dimensions


def _hours(seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    whole = int(seconds)
    return f"{whole // 3600}h {(whole % 3600) // 60:02d}m"


def measure(
    texts: Sequence[str],
    encoder_inputs: Sequence[object],
    *,
    token_lengths: TokenLengths,
    encode: EncodeBatch,
    score: ScoreBatch,
    pending_chunks: int,
    rerank_candidates: int,
    load_seconds: float,
    peak_rss_bytes: int,
    limit: int,
    expected_dimensions: int,
) -> BenchmarkReport:
    """Time the sample and assemble the report.

    ``texts`` are the raw chunk bodies, for the estimator comparison.
    ``encoder_inputs`` are whatever the encoder is actually fed - pairs, for the
    article encoder - so the timing covers the real input shape rather than an
    approximation of it.
    """
    real = token_lengths(encoder_inputs)
    tokens = TokenReport(
        sampled=len(texts),
        estimated_total=sum(estimate_tokens(text) for text in texts),
        real_total=sum(real),
        max_real=max(real) if real else 0,
        truncated=sum(1 for length in real if length > limit),
        limit=limit,
    )

    started = time.monotonic()
    vectors = encode(encoder_inputs)
    encode_seconds = time.monotonic() - started

    query = texts[0] if texts else "adverse reaction"
    candidates = list(texts)[:rerank_candidates]
    started = time.monotonic()
    if candidates:
        score(query, candidates)
    rerank_seconds = time.monotonic() - started

    report = BenchmarkReport(
        sample=len(encoder_inputs),
        load_seconds=load_seconds,
        measured_dimensions=int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        expected_dimensions=expected_dimensions,
        tokens=tokens,
        encode_seconds=encode_seconds,
        pending_chunks=pending_chunks,
        rerank_candidates=len(candidates),
        rerank_seconds=rerank_seconds,
        peak_rss_bytes=peak_rss_bytes,
    )
    log.info(
        "rag.benchmark.measured",
        sample=report.sample,
        chunks_per_second=round(report.chunks_per_second, 3),
        blocked=report.blocked,
    )
    return report


def render(report: BenchmarkReport) -> str:
    """The report as a block of text, ending in a verdict rather than a number."""
    tokens = report.tokens
    lines = [
        f"sample                {report.sample} chunks",
        f"model load            {report.load_seconds:.1f} s",
        f"output width          {report.measured_dimensions} "
        f"(column holds {report.expected_dimensions})",
        "",
        f"tokens, estimate      {tokens.estimated_total:,}",
        f"tokens, tokenizer     {tokens.real_total:,}",
        f"ratio                 {tokens.ratio:.2f} estimated per real token",
        f"longest chunk         {tokens.max_real} tokens (limit {tokens.limit})",
        f"would truncate        {tokens.truncated} of {tokens.sampled}",
        "",
        f"encode                {report.encode_seconds:.1f} s "
        f"for {report.sample} -> {report.chunks_per_second:.2f} chunks/s",
        f"peak memory           {report.peak_rss_bytes / 1024**2:.0f} MiB",
        f"chunks left to embed  {report.pending_chunks:,}",
        f"projected corpus run  {_hours(report.projected_seconds)}",
        "",
        f"rerank                {report.rerank_seconds:.2f} s "
        f"for {report.rerank_candidates} candidates, one query",
        "",
        f"estimator: {tokens.verdict}",
    ]

    if report.measured_dimensions != report.expected_dimensions:
        lines += [
            "",
            "STOP. The model produced "
            f"{report.measured_dimensions}-dimensional vectors and the embedding "
            f"column holds {report.expected_dimensions}. Storing them would need "
            "the vectors truncated or padded, which makes the index something "
            "other than what it reports. Do not start a corpus run.",
        ]
    if tokens.truncated:
        lines += [
            "",
            f"STOP. {tokens.truncated} of {tokens.sampled} sampled chunks exceed "
            f"{tokens.limit} tokens and would be truncated by the encoder. The text "
            "past the cut would be text the index claims to hold and does not. The "
            "chunk target needs revisiting before a multi-hour run, not after one.",
        ]
    if not report.blocked:
        lines += ["", "No blocking condition. A corpus run can start."]
    return "\n".join(lines)
