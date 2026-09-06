"""The run artifact, and the unit loop that feeds it.

The artifact is this prompt's deliverable: the hit rate it records is what
decides whether the scope widens past ``top_k``. So what is tested here is
mostly arithmetic and honesty - that both rates are computed from terminal state
over the whole selected scope rather than over whatever this run happened to
attempt, that N is reported whether or not the cap bit, and that the two things
known to be misleading about this pipeline (the no-op conjunct, the dead
ingredient route) are stated in the document rather than left for a reader to
infer.

Two numbers are pinned here because they have already been wrong once. 0.955 is
the double-counted figure that counted each period-suffixed twin in both
numerator and denominator; it is reachable again by making the slots rate count
slots instead of queries, so a test carries its name. And null is kept distinct
from 0.0, because "nothing was selected" and "nothing resolved" are different
findings.

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


def _unit(
    name: str = "LIPITOR 10MG", route: str = "cleaned_string", query: str | None = None
) -> ScopeUnit:
    """One selected unit. ``query`` defaults to the name, so units built with
    distinct names are distinct queries unless a test deliberately collides
    them - which is what the duplication tests need to do."""
    return ScopeUnit(
        folded_string=name,
        query=query if query is not None else name,
        route=route,
        ingredient_rxcui=83367 if route == "ingredient" else None,
        flagged_pairs=7,
    )


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    # openfda_api_key is pinned empty, not left to the environment. Settings
    # reads the process env, so without this the daily-cap assertions below
    # depend on whether the machine running them happens to have a key: they
    # passed for weeks only because the container had none, and went red the
    # moment one was supplied. A test must not measure its host.
    fields: dict[str, Any] = {
        "django_secret_key": "test",
        "data_dir": tmp_path,
        "cache_dir": tmp_path / "cache",
        "openfda_api_key": "",
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

    def test_the_rates_are_over_the_selected_scope_not_over_this_runs_attempts(
        self, tmp_path: Path
    ) -> None:
        """The denominators are what the cap selected, not what this run tried.

        Four selected units, four distinct queries. One fetched and resolved, one
        fetched and did not, one skipped having attached to an earlier run's
        fetch, one errored. Two of the four queries reach a label, so both
        denominators are 4 and the numerator is 2 - even though only two units
        were attempted.
        """
        document = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("A"), labels=2),
                UnitResult(unit=_unit("B"), labels=0),
                UnitResult(unit=_unit("C"), skipped=True, reason="attached", drug_keys=5),
                UnitResult(unit=_unit("D"), error="boom", terminal_state_known=False),
            ),
            _settings(tmp_path),
        )
        resolution = document["resolution"]
        assert isinstance(resolution, dict)
        assert resolution["attempted"] == 2
        assert resolution["distinct_queries_selected"] == 4
        assert resolution["distinct_queries_resolved"] == 2
        assert resolution["hit_rate_distinct_queries"] == 0.5
        assert resolution["hit_rate_selected_slots"] == 0.5
        assert resolution["skipped_already_complete"] == 1
        assert resolution["errors"] == 1
        # 0.955 was the double-counted figure. A reader offered three rates
        # quotes the wrong one, so the ambiguous key is gone rather than kept.
        assert "hit_rate" not in resolution

    def test_a_warm_re_run_reports_the_scope_and_not_the_one_unit_it_attempted(
        self, tmp_path: Path
    ) -> None:
        """The defect this rule replaced, in miniature.

        The 06:05 re-run attempted a single unit and reported both rates as 1.0
        for a scope of 200 strings, because both denominators came from that
        run's attempts. Here one unit fetches and three attach; the rates must
        describe four slots and three queries, not one.
        """
        document = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("A", query="A"), labels=3),
                UnitResult(unit=_unit("A.", query="A"), skipped=True, drug_keys=3),
                UnitResult(unit=_unit("B", query="B"), skipped=True, drug_keys=1),
                UnitResult(unit=_unit("C", query="C"), skipped=True, reason="nothing"),
            ),
            _settings(tmp_path),
        )
        resolution = document["resolution"]
        assert isinstance(resolution, dict)
        assert resolution["attempted"] == 1
        assert resolution["distinct_queries_selected"] == 3
        assert resolution["distinct_queries_resolved"] == 2
        assert resolution["hit_rate_distinct_queries"] == round(2 / 3, 4)
        assert resolution["hit_rate_selected_slots"] == 0.5
        assert resolution["hit_rate_distinct_queries"] != 1.0
        assert resolution["hit_rate_selected_slots"] != 1.0

    def test_null_only_for_an_empty_scope_and_zero_when_nothing_resolves(
        self, tmp_path: Path
    ) -> None:
        """Two different states that must not collapse onto one another.

        Under the old rule these were the same thing, because a scope of skipped
        units had nothing attempted and so no denominator. Now the denominator is
        the scope, so a scope where nothing reaches a label measures a real 0.0,
        and None is reserved for having selected nothing at all.
        """
        empty = pipeline.artifact(_result(), _settings(tmp_path))["resolution"]
        assert isinstance(empty, dict)
        assert empty["hit_rate_distinct_queries"] is None
        assert empty["hit_rate_selected_slots"] is None

        barren = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("A"), skipped=True, reason="resolved to nothing"),
                UnitResult(unit=_unit("B"), labels=0),
            ),
            _settings(tmp_path),
        )["resolution"]
        assert isinstance(barren, dict)
        assert barren["hit_rate_distinct_queries"] == 0.0
        assert barren["hit_rate_selected_slots"] == 0.0

    def test_the_per_slot_numerator_would_reintroduce_the_retired_95_5(
        self, tmp_path: Path
    ) -> None:
        """The trap, at the real proportions, pinned so it cannot be walked back.

        The measured P05 scope: 200 selected strings over 169 distinct queries,
        160 of which resolve, and 191 of the 200 slots reach a label because the
        9 that do not are all singleton queries. Dividing resolving *slots* by
        slots gives 191/200 = 0.955, which is exactly the double-counted figure
        that grouping by query was introduced to retire.

        The numerator is resolved queries in both rates. Anyone later "fixing"
        hit_rate_selected_slots to count slots reproduces 0.955 and fails here.
        """
        units: list[UnitResult] = []
        # 130 singleton queries that resolve.
        for index in range(130):
            units.append(UnitResult(unit=_unit(f"S{index}", query=f"S{index}"), labels=1))
        # 29 two-member groups that resolve: one fetches, one attaches.
        for index in range(29):
            query = f"P{index}"
            units.append(UnitResult(unit=_unit(f"{query}a", query=query), labels=1))
            units.append(
                UnitResult(unit=_unit(f"{query}b", query=query), skipped=True, drug_keys=1)
            )
        # One three-member group that resolves.
        units.append(UnitResult(unit=_unit("Ta", query="T"), labels=1))
        units.append(UnitResult(unit=_unit("Tb", query="T"), skipped=True, drug_keys=1))
        units.append(UnitResult(unit=_unit("Tc", query="T"), skipped=True, drug_keys=1))
        # 9 singleton queries that resolve to nothing.
        for index in range(9):
            units.append(UnitResult(unit=_unit(f"Z{index}", query=f"Z{index}"), labels=0))

        resolution = pipeline.artifact(_result(*units), _settings(tmp_path))["resolution"]
        assert isinstance(resolution, dict)
        assert resolution["distinct_queries_selected"] == 169
        assert resolution["distinct_queries_resolved"] == 160
        assert len(units) == 200
        assert sum(1 for item in units if item.reaches_label) == 191

        assert resolution["hit_rate_selected_slots"] == 0.80
        assert resolution["hit_rate_selected_slots"] != 0.955
        assert resolution["hit_rate_distinct_queries"] == 0.9467

    def test_an_unreadable_terminal_state_is_counted_rather_than_absorbed(
        self, tmp_path: Path
    ) -> None:
        """An unknown must not read as a miss without saying so."""
        resolution = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("A"), labels=1),
                UnitResult(unit=_unit("B"), skipped=True, terminal_state_known=False),
            ),
            _settings(tmp_path),
        )["resolution"]
        assert isinstance(resolution, dict)
        assert resolution["distinct_queries_unknown_state"] == 1
        assert resolution["distinct_queries_resolved"] == 1
        # In the denominator, out of the numerator: the rate is a lower bound.
        assert resolution["hit_rate_distinct_queries"] == 0.5

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
        assert "every distinct query costs its own request" in note
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

    def test_network_and_cache_costs_are_summed_apart(self, tmp_path: Path) -> None:
        """A warm cache must not read as network cost against the daily cap."""
        document = pipeline.artifact(
            _result(
                UnitResult(
                    unit=_unit("A"),
                    labels=1,
                    requests=2,
                    cache_hits=1,
                    network_bytes=100,
                    cache_bytes=7,
                ),
                UnitResult(
                    unit=_unit("B"),
                    labels=0,
                    requests=1,
                    cache_hits=3,
                    network_bytes=50,
                    cache_bytes=9,
                ),
            ),
            _settings(tmp_path),
        )
        cost = document["cost"]
        assert isinstance(cost, dict)
        assert cost["network_requests"] == 3
        assert cost["cache_hits"] == 4
        assert cost["pages_total"] == 7
        assert cost["network_bytes"] == 150
        assert cost["cache_bytes"] == 16
        # The names that conflated the two are gone, not redefined.
        assert "requests" not in cost
        assert "bytes_received" not in cost

    def test_two_strings_sending_one_query_are_one_distinct_query(self, tmp_path: Path) -> None:
        """The period fold makes twins share a query; they are one drug.

        Grouping is derived from the query sent rather than declared, so a
        future cleaner change is reflected without a second rule to maintain.
        """
        document = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("GABAPENTIN", query="GABAPENTIN"), labels=3),
                UnitResult(unit=_unit("GABAPENTIN.", query="GABAPENTIN"), labels=3),
                UnitResult(unit=_unit("ASPIRIN", query="ASPIRIN"), labels=2),
                UnitResult(unit=_unit("ZOPICLONE", query="ZOPICLONE"), labels=0),
            ),
            _settings(tmp_path),
        )
        resolution = document["resolution"]
        assert isinstance(resolution, dict)
        assert resolution["attempted"] == 4
        assert resolution["distinct_queries_selected"] == 3
        assert resolution["distinct_queries_resolved"] == 2
        assert resolution["hit_rate_distinct_queries"] == round(2 / 3, 4)
        # Same numerator, the slots the cap actually bought as the denominator.
        assert resolution["hit_rate_selected_slots"] == 0.5

    def test_a_three_way_query_group_collapses_to_one(self, tmp_path: Path) -> None:
        """RITUXIMAB, RITUXIMAB. and RITUXIMAB (UNKNOWN) all send one query."""
        document = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("RITUXIMAB", query="RITUXIMAB"), labels=5),
                UnitResult(unit=_unit("RITUXIMAB.", query="RITUXIMAB"), labels=5),
                UnitResult(unit=_unit("RITUXIMAB (UNKNOWN)", query="RITUXIMAB"), labels=5),
            ),
            _settings(tmp_path),
        )
        resolution = document["resolution"]
        assert isinstance(resolution, dict)
        assert resolution["distinct_queries_selected"] == 1
        assert resolution["distinct_queries_resolved"] == 1
        assert resolution["hit_rate_distinct_queries"] == 1.0
        assert resolution["hit_rate_selected_slots"] == round(1 / 3, 4)

    def test_both_rates_are_emitted_and_the_ambiguous_one_is_gone(self, tmp_path: Path) -> None:
        """Neither rate may be quoted alone, so neither may be absent."""
        resolution = pipeline.artifact(
            _result(UnitResult(unit=_unit("A"), labels=1)), _settings(tmp_path)
        )["resolution"]
        assert isinstance(resolution, dict)
        assert "hit_rate_distinct_queries" in resolution
        assert "hit_rate_selected_slots" in resolution
        assert "hit_rate" not in resolution
        assert "duplication cost" in str(resolution["rate_note"])

    def test_stored_reports_key_rows_above_distinct_documents(self, tmp_path: Path) -> None:
        """18048 was a drug-key count read as a document count.

        Two strings reaching the same label write two keys and one document,
        and the ratio between the two figures is the duplication signal.
        """
        document = pipeline.artifact(
            _result(
                UnitResult(
                    unit=_unit("GABAPENTIN", query="GABAPENTIN"),
                    labels=2,
                    sections=5,
                    document_sections={"set-a": 3, "set-b": 2},
                ),
                UnitResult(
                    unit=_unit("GABAPENTIN.", query="GABAPENTIN"),
                    labels=2,
                    sections=5,
                    document_sections={"set-a": 3, "set-b": 2},
                ),
            ),
            _settings(tmp_path),
        )
        stored = document["stored"]
        assert isinstance(stored, dict)
        assert stored["drug_key_rows"] == 4
        assert stored["section_writes"] == 10
        assert stored["distinct_documents"] == 2
        assert stored["distinct_sections"] == 5
        # The names that read as document counts are retired, not redefined.
        assert "documents" not in stored
        assert "sections" not in stored

    def test_every_unit_gets_a_row_carrying_its_own_facts(self, tmp_path: Path) -> None:
        """The page cap is the reason this array is not optional."""
        document = pipeline.artifact(
            _result(
                UnitResult(
                    unit=_unit("IBUPROFEN", query="IBUPROFEN"),
                    labels=1000,
                    sections=2400,
                    requests=10,
                    cache_hits=0,
                    network_bytes=900,
                    cache_bytes=0,
                    page_cap=True,
                ),
                UnitResult(
                    unit=_unit("IBUPROFEN.", query="IBUPROFEN"),
                    labels=1000,
                    sections=2400,
                    requests=0,
                    cache_hits=10,
                    network_bytes=0,
                    cache_bytes=900,
                    page_cap=True,
                ),
                UnitResult(unit=_unit("ZOFRAN", query="ZOFRAN"), labels=0, requests=1),
            ),
            _settings(tmp_path),
        )
        units = document["units"]
        assert isinstance(units, list)
        assert len(units) == 3
        first, second, third = units

        assert first["folded_string"] == "IBUPROFEN"
        assert first["query"] == "IBUPROFEN"
        assert first["route"] == "cleaned_string"
        assert first["documents"] == 1000
        assert first["page_cap"] is True
        assert (first["requests"], first["cache_hits"]) == (10, 0)

        # The twin: same labels, no network. Recoverable per unit, not summed away.
        assert second["folded_string"] == "IBUPROFEN."
        assert (second["requests"], second["cache_hits"]) == (0, 10)
        assert (second["network_bytes"], second["cache_bytes"]) == (0, 900)

        assert third["documents"] == 0
        assert third["page_cap"] is False

    def test_the_unit_rows_follow_selection_order(self, tmp_path: Path) -> None:
        """Selection order is the ranking, and it is deterministic, so two runs
        of the same scope diff cleanly."""
        names = ["PREDNISONE", "METHOTREXATE", "HUMIRA"]
        document = pipeline.artifact(
            _result(*(UnitResult(unit=_unit(n), labels=1) for n in names)),
            _settings(tmp_path),
        )
        units = document["units"]
        assert isinstance(units, list)
        assert [row["folded_string"] for row in units] == names

    def test_a_skipped_or_failed_unit_still_gets_a_row(self, tmp_path: Path) -> None:
        """A unit missing from the array would be invisible rather than explained."""
        document = pipeline.artifact(
            _result(
                UnitResult(unit=_unit("A"), skipped=True, reason="already completed"),
                UnitResult(unit=_unit("B"), error="boom"),
            ),
            _settings(tmp_path),
        )
        units = document["units"]
        assert isinstance(units, list)
        assert len(units) == 2
        assert units[0]["skipped"] is True
        assert units[0]["reason"] == "already completed"
        assert units[1]["error"] == "boom"

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
        # No recorded count on the row, so nothing can be concluded and the
        # decision's own reason stands.
        monkeypatch.setattr(store, "attach_drug_keys", lambda **kwargs: 0)
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
            pipeline, "_fetch", lambda *args: client.SearchResult(results=[], requests=1)
        )
        monkeypatch.setattr(
            store, "store_labels", lambda *args, **kwargs: store.StoreCounts(0, 0, 0)
        )
        result = pipeline.ingest_unit(_unit(), _settings(tmp_path))
        assert not result.hit
        assert not result.error
        assert completed == [_unit().manifest_unit]

    def test_a_shared_query_attaches_keys_without_restoring_documents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The constraint: the skip path writes keys and nothing else.

        Re-storing documents or sections here would put distinct_documents and
        section_writes wrong again, in a new way.
        """
        stored: list[str] = []
        monkeypatch.setattr(
            manifest,
            "decide",
            lambda unit, force=False: manifest.Decision(False, "already completed", 4),
        )
        monkeypatch.setattr(store, "attach_drug_keys", lambda **kwargs: 4)
        monkeypatch.setattr(
            store, "store_labels", lambda *a, **k: stored.append("documents") or None
        )
        monkeypatch.setattr(pipeline, "_fetch", lambda *a: pytest.fail("must not fetch"))

        result = pipeline.ingest_unit(_unit("GABAPENTIN."), _settings(tmp_path))

        assert result.skipped is True
        assert result.drug_keys == 4
        assert result.documents == 0 if hasattr(result, "documents") else result.labels == 0
        assert result.sections == 0
        assert stored == [], "store_labels must not run on the attach path"
        assert result.reason == "query already fetched; drug keys attached"
        assert not result.error

    def test_an_attached_unit_carries_the_page_cap_of_the_fetch_it_attached_to(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It did not page, so it cannot measure truncation. It must not guess.

        P12 excludes page-capped strings by reading this out of the artifact, so
        a false written by a unit that never paged puts truncated labels into a
        corpus that claims to have none. In the 06:05 artifact that was 190 rows
        asserting no truncation without measuring it.
        """
        monkeypatch.setattr(
            manifest,
            "decide",
            lambda unit, force=False: manifest.Decision(
                False, "already completed", 4, page_capped=True
            ),
        )
        monkeypatch.setattr(store, "attach_drug_keys", lambda **kwargs: 4)
        monkeypatch.setattr(pipeline, "_fetch", lambda *a: pytest.fail("must not fetch"))

        result = pipeline.ingest_unit(_unit("IBUPROFEN."), _settings(tmp_path))

        assert result.skipped is True
        assert result.page_cap is True

    def test_an_attached_unit_reports_null_when_the_row_never_recorded_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Null is not false, and the artifact has to keep them apart.

        False is a measurement that truncation did not happen. Null is the
        absence of one. A consumer excluding page-capped strings must exclude
        null too, and it can only do that if null survives to the document.
        """
        monkeypatch.setattr(
            manifest,
            "decide",
            lambda unit, force=False: manifest.Decision(
                False, "already completed", 4, page_capped=None
            ),
        )
        monkeypatch.setattr(store, "attach_drug_keys", lambda **kwargs: 4)
        monkeypatch.setattr(pipeline, "_fetch", lambda *a: pytest.fail("must not fetch"))

        result = pipeline.ingest_unit(_unit("LEGACY"), _settings(tmp_path))
        assert result.page_cap is None
        assert result.page_cap is not False

        row = pipeline._unit_row(result)
        assert row["page_cap"] is None

    def test_a_query_that_resolved_to_nothing_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing to attach because there was nothing. Normal, and named."""
        monkeypatch.setattr(
            manifest,
            "decide",
            lambda unit, force=False: manifest.Decision(False, "already completed", 0),
        )
        monkeypatch.setattr(store, "attach_drug_keys", lambda **kwargs: 0)
        result = pipeline.ingest_unit(_unit("ZOFRAN"), _settings(tmp_path))
        assert result.skipped is True
        assert result.drug_keys == 0
        assert result.reason == "query already fetched and resolved to nothing"
        assert not result.error

    def test_a_manifest_claiming_documents_with_no_keys_is_an_error_not_a_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The state that must be impossible to mistake for a normal skip.

        The manifest says the fetch stored documents, but nothing can be found
        to attach to. That is an inconsistency, not an outcome, so it is
        recorded as an error and counted as one.
        """
        monkeypatch.setattr(
            manifest,
            "decide",
            lambda unit, force=False: manifest.Decision(False, "already completed", 12),
        )
        monkeypatch.setattr(store, "attach_drug_keys", lambda **kwargs: 0)
        result = pipeline.ingest_unit(_unit("GABAPENTIN."), _settings(tmp_path))

        assert result.skipped is False, "an inconsistency must not read as a skip"
        assert result.drug_keys == 0
        assert "12 documents" in result.error
        assert "no drug keys were found" in result.error
        assert result.reason == ""

    def test_either_member_of_a_query_group_may_fetch_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Group members are not adjacent in selection order and the suffixed
        form sorts first in 2 of 30 groups, so neither order may be assumed."""
        completed: set[str] = set()
        attached: list[str] = []
        monkeypatch.setattr(
            manifest,
            "decide",
            lambda unit, force=False: (
                manifest.Decision(False, "already completed", 3)
                if unit in completed
                else manifest.Decision(True, "not ingested")
            ),
        )
        monkeypatch.setattr(manifest, "start", lambda unit: None)
        monkeypatch.setattr(manifest, "complete", lambda unit, **k: completed.add(unit))
        monkeypatch.setattr(
            store, "attach_drug_keys", lambda **k: attached.append(k["folded_string"]) or 3
        )
        monkeypatch.setattr(
            pipeline, "_fetch", lambda *a: client.SearchResult(results=[], requests=1)
        )
        monkeypatch.setattr(store, "store_labels", lambda *a, **k: store.StoreCounts(3, 6, 3))

        bare = _unit("GABAPENTIN", query="GABAPENTIN")
        suffixed = _unit("GABAPENTIN.", query="GABAPENTIN")
        # The suffixed form first, which is the order 2 of 30 real groups take.
        first = pipeline.ingest_unit(suffixed, _settings(tmp_path))
        second = pipeline.ingest_unit(bare, _settings(tmp_path))

        assert first.labels == 3 and not first.skipped, "whoever is first fetches"
        assert second.skipped and second.drug_keys == 3, "the later one attaches"
        assert attached == ["GABAPENTIN"]

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
                requests=1,
                network_bytes=42,
            ),
        )
        monkeypatch.setattr(
            store, "store_labels", lambda *args, **kwargs: store.StoreCounts(1, 1, 1)
        )
        result = pipeline.ingest_unit(_unit(), _settings(tmp_path))
        assert result.hit
        assert result.labels == 1
        assert result.sections == 1
        assert result.network_bytes == 42

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
        pipeline._fetch(_unit(route="cleaned_string", query="LIPITOR"), _settings(tmp_path))
        assert seen == ["rxcui:83367", "name:LIPITOR"]


def test_a_section_record_survives_the_round_trip() -> None:
    """Guards the parse-to-store contract the pipeline depends on."""
    record = LabelRecord(
        set_id="s",
        sections=[LabelSectionRecord(section_code="adverse_reactions", ordinal=0, text="X")],
    )
    assert record.has_sections
    assert record.sections[0].text == "X"
