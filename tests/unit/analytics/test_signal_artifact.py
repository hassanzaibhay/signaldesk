"""The run record and the committed artifact built from it.

This path had two defects that only surfaced when it was run for real: the
history directory resolved to the filesystem root, and the commit was recorded
as ``unknown`` because the container has no ``.git``. Both are pinned here so
the next one surfaces in a second rather than after a seven-minute corpus pass.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from signaldesk.analytics.signals import (
    SignalRun,
    _code_sha,
    _ic_divergence,
    _is_provisional,
    read_record,
    run_root,
    write_history_artifact,
)
from signaldesk.core.config import Settings
from signaldesk.stats.panel import estimate_all
from signaldesk.stats.types import Contingency, MgpsHyperparameters

pytestmark = pytest.mark.unit

PRIOR = MgpsHyperparameters(
    alpha1=2.94073934,
    beta1=0.38537165,
    alpha2=2.15410622,
    beta2=2.03677734,
    p=0.0720271218954894,
    neg_log_likelihood=0.0,
    iterations=0,
    n_pairs=0,
)


def _record(run_id: str, *, provisional: bool = False) -> SignalRun:
    return SignalRun(
        run_id=run_id,
        created_at="2026-08-15T14:17:16+00:00",
        params={"drug_key": "raw_string", "role_codes": "PS,SS", "min_a": 3},
        code_sha="9583b8018ba55c95b5dc28b34b83611f8fe59dfe",
        data_snapshot={"2012Q4": "abc123"},
        n_cases=16_054_992,
        pairs_observed=8_583_614,
        pairs_sufficient=2_785_896,
        duplicates_removed=4_479_514,
        cases_without_drug=0,
        cases_without_reaction=0,
        hyperparameters={"alpha1": 1e-06, "on_boundary": ["alpha1"] if provisional else []},
        mgps_provisional=provisional,
        flag_counts={"ror": 1_693_733, "all_four": None if provisional else 1},
        ic_divergence={"max_absolute": 22.5, "median_absolute": 1.5, "share_above_one": 0.58},
        seconds=433.36,
        peak_rss_bytes=8_759_873_536,
    )


def _write_record(record: SignalRun, settings: Settings) -> None:
    partition = run_root(settings) / f"run={record.run_id}"
    partition.mkdir(parents=True, exist_ok=True)
    (partition / "record.json").write_text(
        json.dumps(record.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def test_the_code_sha_is_taken_from_the_environment_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The container has no .git, so the Makefile passes the host's answer in.

    Without this the artifact recorded ``unknown`` for every run, which is a
    provenance hole that looks like a value only if nobody reads it.
    """
    monkeypatch.setenv("SIGNALDESK_CODE_SHA", "deadbeef")
    assert _code_sha() == "deadbeef"


def test_a_blank_environment_variable_does_not_masquerade_as_a_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIGNALDESK_CODE_SHA", "   ")
    assert _code_sha() != ""


def test_the_artifact_round_trips_both_runs(tmp_path: Path, settings: Settings) -> None:
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    for run_id in ("20260815T141716Z", "20260815T142636Z"):
        _write_record(_record(run_id), scoped)

    path = write_history_artifact(
        ["20260815T141716Z", "20260815T142636Z"], scoped, root=tmp_path / "history"
    )
    document = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path / "history"
    assert [run["run_id"] for run in document["runs"]] == [
        "20260815T141716Z",
        "20260815T142636Z",
    ]
    assert document["runs"][0]["code_sha"] == "9583b8018ba55c95b5dc28b34b83611f8fe59dfe"
    assert document["runs"][0]["population"]["deduplicated_cases"] == 16_054_992


def test_the_artifact_records_that_validation_is_blocked(
    tmp_path: Path, settings: Settings
) -> None:
    """An artifact with no AUROC must say why, not simply omit the field."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_record(_record("20260815T141716Z"), scoped)

    path = write_history_artifact(["20260815T141716Z"], scoped, root=tmp_path / "history")
    blocked = json.loads(path.read_text(encoding="utf-8"))["reference_validation"]

    assert blocked["status"] == "blocked"
    assert len(blocked["required_files"]) == 4
    assert "curated by hand" in blocked["reason"]


def test_a_provisional_run_withholds_mgps_from_the_quotable_list(
    tmp_path: Path, settings: Settings
) -> None:
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_record(_record("20260815T141716Z", provisional=True), scoped)

    path = write_history_artifact(["20260815T141716Z"], scoped, root=tmp_path / "history")
    quotable = json.loads(path.read_text(encoding="utf-8"))["quotable"]

    assert quotable["withheld"] == ["mgps"]
    assert quotable["estimators"] == ["ror", "prr", "bcpnn"]
    assert quotable["withheld_reason"] is not None


def test_an_interior_run_withholds_nothing(tmp_path: Path, settings: Settings) -> None:
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_record(_record("20260815T141716Z", provisional=False), scoped)

    path = write_history_artifact(["20260815T141716Z"], scoped, root=tmp_path / "history")
    quotable = json.loads(path.read_text(encoding="utf-8"))["quotable"]

    assert quotable["withheld"] == []
    assert quotable["withheld_reason"] is None


def test_a_diagnostic_file_is_embedded_when_supplied(tmp_path: Path, settings: Settings) -> None:
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_record(_record("20260815T141716Z", provisional=True), scoped)
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(json.dumps({"verdict": "boundary is the optimum"}), encoding="utf-8")

    path = write_history_artifact(
        ["20260815T141716Z"], scoped, root=tmp_path / "history", diagnostics=diagnostic
    )
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["mgps_boundary_diagnostic"] == {"verdict": "boundary is the optimum"}


def test_no_diagnostic_leaves_the_field_null(tmp_path: Path, settings: Settings) -> None:
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_record(_record("20260815T141716Z"), scoped)

    path = write_history_artifact(["20260815T141716Z"], scoped, root=tmp_path / "history")

    assert json.loads(path.read_text(encoding="utf-8"))["mgps_boundary_diagnostic"] is None


def test_an_empty_run_list_is_refused(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(ValueError, match="at least one run"):
        write_history_artifact([], settings, root=tmp_path)


def test_a_missing_record_names_its_path(tmp_path: Path, settings: Settings) -> None:
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    with pytest.raises(FileNotFoundError, match=r"record\.json"):
        read_record("20990101T000000Z", scoped)


def test_provisional_detection_reads_the_nested_flag() -> None:
    assert _is_provisional({"mgps": {"provisional": True}})
    assert not _is_provisional({"mgps": {"provisional": False}})
    assert not _is_provisional({"mgps": "not a mapping"})
    assert not _is_provisional({})


def test_ic_divergence_ignores_the_undefined_rows() -> None:
    """``a = 0`` makes the unshrunk column -inf, which must not poison the summary.

    A mean taken over an infinity is an infinity, and a divergence reported as
    inf says nothing about the corpus.
    """
    table = Contingency(
        a=np.array([0, 25, 100], dtype=np.int64),
        b=np.array([500, 1_975, 100], dtype=np.int64),
        c=np.array([300, 1_000, 100], dtype=np.int64),
        d=np.array([99_200, 97_000, 99_700], dtype=np.int64),
    )
    divergence = _ic_divergence(estimate_all(table, hyperparameters=PRIOR))

    assert np.isfinite(divergence["max_absolute"])
    assert np.isfinite(divergence["median_absolute"])
    assert 0.0 <= divergence["share_above_one"] <= 1.0


def test_ic_divergence_on_an_all_undefined_table_is_zero_not_nan() -> None:
    table = Contingency(
        a=np.array([0], dtype=np.int64),
        b=np.array([500], dtype=np.int64),
        c=np.array([300], dtype=np.int64),
        d=np.array([99_200], dtype=np.int64),
    )
    divergence = _ic_divergence(estimate_all(table, hyperparameters=PRIOR))

    assert divergence == {"max_absolute": 0.0, "median_absolute": 0.0, "share_above_one": 0.0}


def _diagnostic(tmp_path: Path, **overrides: object) -> Path:
    """A minimal boundary diagnostic, shaped like the real one."""
    document: dict[str, object] = {
        "bound_is_optimum": True,
        "relative_spread": 0.178481939,
        "nll_at_bound": 16_580_220.18,
        "grid_argmin": {"alpha1": 1e-05, "nll": 16_580_224.58},
    }
    document.update(overrides)
    path = tmp_path / "diagnostic.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_withheld_reason_takes_its_spread_from_the_diagnostic(
    tmp_path: Path, settings: Settings
) -> None:
    """The figure was hardcoded, so every artifact repeated one run's number."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_record(_record("20260815T141716Z", provisional=True), scoped)

    path = write_history_artifact(
        ["20260815T141716Z"],
        scoped,
        root=tmp_path / "history",
        diagnostics=_diagnostic(tmp_path, relative_spread=0.25),
    )
    reason = json.loads(path.read_text(encoding="utf-8"))["quotable"]["withheld_reason"]

    assert "25.0 percent" in reason
    assert "17.9" not in reason


def test_a_defective_fit_is_not_described_as_a_genuine_optimum(
    tmp_path: Path, settings: Settings
) -> None:
    """A grid point below the bound means the fit is wrong, not that data is odd."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_record(_record("20260815T141716Z", provisional=True), scoped)

    path = write_history_artifact(
        ["20260815T141716Z"],
        scoped,
        root=tmp_path / "history",
        diagnostics=_diagnostic(tmp_path, bound_is_optimum=False),
    )
    reason = json.loads(path.read_text(encoding="utf-8"))["quotable"]["withheld_reason"]

    assert "genuine optimum" not in reason
    assert "defective" in reason


def test_without_a_diagnostic_the_reason_states_no_number(
    tmp_path: Path, settings: Settings
) -> None:
    """An unmeasured sensitivity must read as unmeasured, not as a figure."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_record(_record("20260815T141716Z", provisional=True), scoped)

    path = write_history_artifact(["20260815T141716Z"], scoped, root=tmp_path / "history")
    reason = json.loads(path.read_text(encoding="utf-8"))["quotable"]["withheld_reason"]

    assert "has not been measured" in reason
    # "EBGM05" is a column name, not a measurement. What must be absent is a
    # reported figure, which is the thing that used to be hardcoded here.
    assert "percent" not in reason


def test_the_writer_sha_is_reported_apart_from_the_runs(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer's commit is not the provenance of the numbers it collects."""
    monkeypatch.setenv("SIGNALDESK_CODE_SHA", "deadbeef")
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_record(_record("20260815T141716Z"), scoped)

    path = write_history_artifact(["20260815T141716Z"], scoped, root=tmp_path / "history")
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["writer_code_sha"] == "deadbeef"
    assert document["runs_code_sha"]["distinct"] == ["9583b8018ba55c95b5dc28b34b83611f8fe59dfe"]
    assert document["runs_code_sha"]["runs_at_unrecorded_commit"] == []
    assert "code_sha" not in document


def test_a_run_at_an_unrecorded_commit_says_so(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the writer's sha reads as provenance the run does not have."""
    monkeypatch.setenv("SIGNALDESK_CODE_SHA", "deadbeef")
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    record = _record("20260815T141716Z")
    _write_record(
        SignalRun(
            **{**{f.name: getattr(record, f.name) for f in fields(record)}, "code_sha": "unknown"}
        ),
        scoped,
    )

    path = write_history_artifact(["20260815T141716Z"], scoped, root=tmp_path / "history")
    summary = json.loads(path.read_text(encoding="utf-8"))["runs_code_sha"]

    assert summary["runs_at_unrecorded_commit"] == ["20260815T141716Z"]
    assert "unrecorded commit" in summary["note"]


def test_a_legacy_flag_count_key_is_renamed_when_the_record_is_read(
    tmp_path: Path, settings: Settings
) -> None:
    """Pre-rename runs are read through an alias rather than rebuilt."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    record = _record("20260815T141716Z")
    legacy = record.as_dict()
    legacy["flag_counts"] = {"three_of_four": 1_393_815, "bcpnn": 1_393_815}
    partition = run_root(scoped) / "run=20260815T141716Z"
    partition.mkdir(parents=True, exist_ok=True)
    (partition / "record.json").write_text(json.dumps(legacy), encoding="utf-8")

    counts = read_record("20260815T141716Z", scoped)["flag_counts"]

    assert counts == {"ror_prr_bcpnn": 1_393_815, "bcpnn": 1_393_815}
