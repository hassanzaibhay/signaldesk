"""The run artifact, and the unit loop that feeds it.

The artifact is this prompt's deliverable: the hit rate it records is what
decides whether the scope widens past ``top_k``. So what is tested here is
mostly arithmetic and honesty - that the rate is computed over the units the
query strategy was actually tried on, that N is reported whether or not the cap
bit, and that the two things known to be misleading about this pipeline (the
no-op conjunct, the dead ingredient route) are stated in the document rather
than left for a reader to infer.

The database and the network are monkeypatched out. What they would do is
covered by ``tests/integration/test_spl_store.py`` and
``tests/unit/ingest/test_spl_client.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from signaldesk.core.config import Settings
from signaldesk.ingest.spl import client, manifest, pipeline, store
from signaldesk.ingest.spl.parse import LabelRecord, LabelSectionRecord
from signaldesk.ingest.spl.pipeline import RunResult, UnitResult
from signaldesk.ingest.spl.scope import ScopeUnit

pytestmark = pytest.mark.unit


def _unit(name: str = "LIPITOR 10MG", route: str = "cleaned_string") -> ScopeUnit:
    return ScopeUnit(
        folded_string=name,
        query="LIPITOR",
        route=route,
        ingredient_rxcui=83367 if route == "ingredient" else None,
        flagged_pairs=7,
    )


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    fields: dict[str, Any] = {
        "django_secret_key": "test",
        "data_dir": tmp_path,
        "cache_dir": tmp_path / "cache",
    }
    fields.update(overrides)
    return Settings(**fields)


def _result(*units: UnitResult, n_total: int = 4000, top_k: int = 200) -> RunResult:
    return RunResult(run_id="20260830T120000Z", n_total=n_total, top_k=top_k, units=list(units))


class TestTheArtifact:
    def test_n_is_the_whole_population_and_the_cap_is_stated_beside_it(
        self, tmp_path: Path
    ) -> None:
        """A reader has to be able to see what fraction of the population ran."""
        document = pipeline.artifact(
            _result(UnitResult(unit=_unit(), labels=1), n_total=4000, top_k=200),
            _settings(tmp_path),
        )
        scope_block = document["scope"]
        assert isinstance(scope_block, dict)
        assert scope_block["n_total_flagged_strings"] == 4000
        assert scope_block["top_k"] == 200
        assert scope_block["selected"] == 1

    def test_the_hit_rate_counts_only_attempted_units(self, tmp_path: Path) -> None:
        """A unit skipped as already complete says nothing about the strategy."""
        document = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("A"), labels=2),
                UnitResult(unit=_unit("B"), labels=0),
                UnitResult(unit=_unit("C"), skipped=True, reason="already completed"),
                UnitResult(unit=_unit("D"), error="boom"),
            ),
            _settings(tmp_path),
        )
        resolution = document["resolution"]
        assert isinstance(resolution, dict)
        assert resolution["attempted"] == 2
        assert resolution["hit_at_least_one_label"] == 1
        assert resolution["resolved_to_nothing"] == 1
        assert resolution["hit_rate"] == 0.5
        assert resolution["skipped_already_complete"] == 1
        assert resolution["errors"] == 1

    def test_the_hit_rate_is_null_rather_than_zero_when_nothing_was_attempted(
        self, tmp_path: Path
    ) -> None:
        """Zero would read as a measured failure of the query strategy."""
        document = pipeline.artifact(
            _result(UnitResult(unit=_unit(), skipped=True)), _settings(tmp_path)
        )
        resolution = document["resolution"]
        assert isinstance(resolution, dict)
        assert resolution["hit_rate"] is None

    def test_units_are_counted_by_route(self, tmp_path: Path) -> None:
        document = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("A", route="cleaned_string"), labels=1),
                UnitResult(unit=_unit("B", route="override"), labels=1),
            ),
            _settings(tmp_path),
        )
        resolution = document["resolution"]
        assert isinstance(resolution, dict)
        assert resolution["by_route"] == {
            "override": 1,
            "ingredient": 0,
            "cleaned_string": 1,
        }
        assert resolution["from_override_file"] == 1

    def test_the_no_op_conjunct_is_disclosed_with_its_counts(self, tmp_path: Path) -> None:
        """It must never be published as a filter that excluded pairs."""
        document = pipeline.artifact(_result(), _settings(tmp_path))
        scope_block = document["scope"]
        assert isinstance(scope_block, dict)
        note = str(scope_block["predicate_note"])
        assert "1393815" in note
        assert "reduces" in note
        assert "must not be reported as a filter" in note

    def test_the_dead_ingredient_route_is_disclosed_in_the_cost_block(self, tmp_path: Path) -> None:
        """Every cost here is a route-2 cost; the artifact has to say so."""
        document = pipeline.artifact(_result(), _settings(tmp_path))
        cost = document["cost"]
        assert isinstance(cost, dict)
        note = str(cost["note"])
        assert "every string costs its own request" in note
        assert "DrugStringMatch is empty" in note

    def test_the_assumed_daily_cap_follows_the_api_key(self, tmp_path: Path) -> None:
        without = pipeline.artifact(_result(), _settings(tmp_path))["cost"]
        with_key = pipeline.artifact(_result(), _settings(tmp_path, openfda_api_key="k"))["cost"]
        assert isinstance(without, dict)
        assert isinstance(with_key, dict)
        assert without["daily_cap_assumed"] == client.DAILY_CAP_WITHOUT_KEY
        assert without["api_key_present"] is False
        assert with_key["daily_cap_assumed"] == client.DAILY_CAP_WITH_KEY
        assert with_key["api_key_present"] is True

    def test_requests_and_bytes_are_summed_across_units(self, tmp_path: Path) -> None:
        document = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("A"), labels=1, requests=2, bytes_received=100),
                UnitResult(unit=_unit("B"), labels=0, requests=1, bytes_received=50),
            ),
            _settings(tmp_path),
        )
        cost = document["cost"]
        assert isinstance(cost, dict)
        assert cost["requests"] == 3
        assert cost["bytes_received"] == 150

    def test_the_artifact_supersedes_the_untracked_head_drugs_figure(self, tmp_path: Path) -> None:
        document = pipeline.artifact(_result(), _settings(tmp_path))
        quotable = document["quotable"]
        assert isinstance(quotable, dict)
        assert "head drugs" in str(quotable["note"])


class TestWritingTheArtifact:
    def test_it_is_written_as_sorted_json_named_for_the_run(self, tmp_path: Path) -> None:
        result = _result(UnitResult(unit=_unit(), labels=1))
        path = pipeline.write_artifact(result, root=tmp_path)
        assert path.name == "spl_ingest_20260830T120000Z.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["scope"]["n_total_flagged_strings"] == 4000
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_the_history_root_is_the_repository_not_the_data_volume(self) -> None:
        """Artifacts are committed, so they cannot live under a Docker volume."""
        assert pipeline.history_root().parts[-2:] == ("evals", "history")
        assert pipeline.history_root().is_dir()


class TestTheChecksum:
    def test_it_changes_when_a_label_is_revised(self) -> None:
        """The manifest is how a reviewer sees that stored content moved."""
        before = LabelRecord(set_id="s", version="1", effective_time="20240101")
        after = LabelRecord(set_id="s", version="2", effective_time="20250101")
        assert pipeline._checksum([before]) != pipeline._checksum([after])

    def test_it_is_stable_for_the_same_content(self) -> None:
        record = LabelRecord(set_id="s", version="1", effective_time="20240101")
        assert pipeline._checksum([record]) == pipeline._checksum([record])


class TestTheUnitLoop:
    def test_a_completed_unit_is_skipped_without_fetching(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            manifest,
            "decide",
            lambda unit, force=False: manifest.Decision(False, "already completed"),
        )
        monkeypatch.setattr(
            pipeline, "_fetch", lambda *args: pytest.fail("must not fetch a completed unit")
        )
        result = pipeline.ingest_unit(_unit(), _settings(tmp_path))
        assert result.skipped
        assert result.reason == "already completed"
        assert not result.hit

    def test_an_empty_query_is_an_error_and_never_reaches_the_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cleaner can return nothing; asking openFDA for nothing is a waste."""
        monkeypatch.setattr(
            manifest, "decide", lambda unit, force=False: manifest.Decision(True, "not ingested")
        )
        monkeypatch.setattr(
            pipeline, "_fetch", lambda *args: pytest.fail("must not fetch an empty query")
        )
        blank = ScopeUnit(
            folded_string="   ",
            query="",
            route="cleaned_string",
            ingredient_rxcui=None,
            flagged_pairs=1,
        )
        result = pipeline.ingest_unit(blank, _settings(tmp_path))
        assert result.error
        assert not result.hit

    def test_a_miss_is_recorded_as_a_completed_unit_with_no_labels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drug with no label is a measurement, not a failure to retry forever."""
        completed: list[str] = []
        monkeypatch.setattr(
            manifest, "decide", lambda unit, force=False: manifest.Decision(True, "not ingested")
        )
        monkeypatch.setattr(manifest, "start", lambda unit: None)
        monkeypatch.setattr(manifest, "complete", lambda unit, **kwargs: completed.append(unit))
        monkeypatch.setattr(
            pipeline, "_fetch", lambda *args: client.SearchResult(results=[], requests_made=1)
        )
        monkeypatch.setattr(
            store, "store_labels", lambda *args, **kwargs: store.StoreCounts(0, 0, 0)
        )
        result = pipeline.ingest_unit(_unit(), _settings(tmp_path))
        assert not result.hit
        assert not result.error
        assert completed == [_unit().manifest_unit]

    def test_a_failure_is_recorded_and_does_not_end_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One awkward string must not abandon the hundreds behind it."""
        failures: list[str] = []
        monkeypatch.setattr(
            manifest, "decide", lambda unit, force=False: manifest.Decision(True, "not ingested")
        )
        monkeypatch.setattr(manifest, "start", lambda unit: None)
        monkeypatch.setattr(manifest, "fail", lambda unit, error: failures.append(unit))

        def _boom(*args: Any) -> client.SearchResult:
            raise client.SplRequestError("openFDA returned 500")

        monkeypatch.setattr(pipeline, "_fetch", _boom)
        result = pipeline.ingest_unit(_unit(), _settings(tmp_path))
        assert "SplRequestError" in result.error
        assert failures == [_unit().manifest_unit]

    def test_a_hit_records_what_was_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            manifest, "decide", lambda unit, force=False: manifest.Decision(True, "not ingested")
        )
        monkeypatch.setattr(manifest, "start", lambda unit: None)
        monkeypatch.setattr(manifest, "complete", lambda unit, **kwargs: None)
        monkeypatch.setattr(
            pipeline,
            "_fetch",
            lambda *args: client.SearchResult(
                results=[{"set_id": "s", "adverse_reactions": ["Headache."]}],
                requests_made=1,
                bytes_received=42,
            ),
        )
        monkeypatch.setattr(
            store, "store_labels", lambda *args, **kwargs: store.StoreCounts(1, 1, 1)
        )
        result = pipeline.ingest_unit(_unit(), _settings(tmp_path))
        assert result.hit
        assert result.labels == 1
        assert result.sections == 1
        assert result.bytes_received == 42

    def test_the_ingredient_route_queries_by_rxcui(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dead in practice, but the branch must still route correctly."""
        seen: list[str] = []
        monkeypatch.setattr(
            client,
            "by_rxcui",
            lambda rxcui, settings: seen.append(f"rxcui:{rxcui}") or client.SearchResult(),
        )
        monkeypatch.setattr(
            client,
            "by_brand_name",
            lambda name, settings: seen.append(f"name:{name}") or client.SearchResult(),
        )
        pipeline._fetch(_unit(route="ingredient"), _settings(tmp_path))
        pipeline._fetch(_unit(route="cleaned_string"), _settings(tmp_path))
        assert seen == ["rxcui:83367", "name:LIPITOR"]


def test_a_section_record_survives_the_round_trip() -> None:
    """Guards the parse-to-store contract the pipeline depends on."""
    record = LabelRecord(
        set_id="s",
        sections=[LabelSectionRecord(section_code="adverse_reactions", ordinal=0, text="X")],
    )
    assert record.has_sections
    assert record.sections[0].text == "X"
