"""The signal-carrying scope: its predicate, its cap, and the query cleaner."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from signaldesk.analytics import signals
from signaldesk.core.config import Settings
from signaldesk.ingest.spl import scope
from signaldesk.ingest.spl.scope import ScopeUnit, SignalScopeError

pytestmark = pytest.mark.unit


def _settings(tmp_path: Path) -> Settings:
    return Settings(django_secret_key="test", data_dir=tmp_path, cache_dir=tmp_path / "cache")


def _write_signal(tmp_path: Path, frame: pl.DataFrame, run_id: str = "20260101T000000Z") -> None:
    # The partition name belongs to analytics/signals.py:364, which is the
    # writer. Building the fixture through signal_root pins it to that writer
    # rather than restating a layout here: if the layout moves, these tests
    # move with it instead of staying green against a layout nothing produces.
    partition = signals.signal_root(_settings(tmp_path)) / f"run={run_id}"
    partition.mkdir(parents=True)
    frame.write_parquet(partition / "part-0.parquet")


def _frame(rows: list[tuple[str, str, bool, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "drug": [row[0] for row in rows],
            "pt": [row[1] for row in rows],
            "flag_ror_prr_bcpnn": [row[2] for row in rows],
            "insufficient": [row[3] for row in rows],
        }
    )


def _legacy_frame(rows: list[tuple[str, str, bool, bool]]) -> pl.DataFrame:
    """A run written before flag_three_of_four was renamed. See signals.py:498."""
    return _frame(rows).rename({"flag_ror_prr_bcpnn": "flag_three_of_four"})


class TestTheMissingTableIsAnErrorNotAnEmptyScope:
    """A rebuild in progress must not be recorded as a measurement of zero."""

    def test_absent_directory_names_the_path(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        with pytest.raises(SignalScopeError) as caught:
            scope.select(settings=settings)
        message = str(caught.value)
        assert str(tmp_path / "parquet" / "signal") in message
        assert "not a scope of zero drugs" in message

    def test_present_but_empty_directory_is_also_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "parquet" / "signal").mkdir(parents=True)
        with pytest.raises(SignalScopeError, match="no signal run partition") as caught:
            scope.select(settings=_settings(tmp_path))
        assert "it is empty" in str(caught.value)

    def test_a_named_run_that_does_not_exist_is_an_error(self, tmp_path: Path) -> None:
        _write_signal(tmp_path, _frame([("A", "X", True, False)]))
        with pytest.raises(SignalScopeError, match="could not read signal run") as caught:
            scope.select("20990101T000000Z", settings=_settings(tmp_path))
        # The read is delegated, so the underlying failure is kept rather than
        # summarised into scope's own wording.
        assert isinstance(caught.value.__cause__, FileNotFoundError)

    def test_a_frame_missing_the_predicate_columns_is_an_error(self, tmp_path: Path) -> None:
        _write_signal(tmp_path, pl.DataFrame({"drug": ["A"], "pt": ["X"]}))
        with pytest.raises(SignalScopeError, match="lacks"):
            scope.select(settings=_settings(tmp_path))


class TestTheSignalPartitionLayoutIsNotRedefinedHere:
    """scope reads the layout through analytics.signals rather than restating it.

    scope.py carried its own copy of the partition layout and the copy drifted:
    it looked for ``run_id=`` while the build wrote ``run=``. Nothing caught it
    because the fixture helper carried the same copy, so the suite agreed with
    the bug.

    The coupling to analytics.signals lives in these tests on purpose. scope.py
    now names no partition at all, which means nothing inside it can assert the
    layout is right; only a test that writes the way the build writes can.
    """

    def test_a_partition_named_the_way_the_build_names_it_is_found(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _write_signal(tmp_path, _frame([("LIPITOR 10MG", "X", True, False)]))
        assert scope.latest_run_id(settings) == "20260101T000000Z"
        # Both sides of the contract, read through their own public entry
        # points. If either moves without the other, this diverges.
        assert scope.latest_run_id(settings) == signals.latest_run(settings)
        _, n_total = scope.select(settings=settings, overrides={}, rxcuis={})
        assert n_total == 1

    def test_a_non_empty_root_is_not_reported_as_a_build_that_wrote_nothing(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "parquet" / "signal"
        # The decoy is the name scope used to look for. A reader in the wrong
        # place and a build that wrote nothing produce the same empty result,
        # and the message must not pick between them.
        (root / "run_id=20260101T000000Z").mkdir(parents=True)
        with pytest.raises(SignalScopeError) as caught:
            scope.select(settings=_settings(tmp_path), overrides={}, rxcuis={})
        message = str(caught.value)
        assert "wrote nothing" not in message
        assert "run_id=20260101T000000Z" in message
        assert str(root) in message

    def test_a_long_listing_announces_its_own_truncation(self, tmp_path: Path) -> None:
        """A diagnostic that silently truncates defeats the diagnostic."""
        root = tmp_path / "parquet" / "signal"
        root.mkdir(parents=True)
        for index in range(12):
            (root / f"decoy-{index:02d}").mkdir()
        with pytest.raises(SignalScopeError) as caught:
            scope.select(settings=_settings(tmp_path), overrides={}, rxcuis={})
        message = str(caught.value)
        assert "12 entries, first 10:" in message
        assert "wrote nothing" not in message

    def test_a_root_that_cannot_be_listed_does_not_bury_the_error(self, tmp_path: Path) -> None:
        """The listing runs while reporting a failure and must not replace it.

        A file where a directory belongs is the cheapest way to make iterdir
        raise OSError without mocking.
        """
        not_a_directory = tmp_path / "signal-is-a-file"
        not_a_directory.write_text("", encoding="utf-8")
        assert scope._describe_contents(not_a_directory) == "its contents could not be listed"

    def test_a_pre_rename_run_is_read_through_the_owning_module(self, tmp_path: Path) -> None:
        """Delegation fixed a second defect: scope could not read a legacy run.

        analytics.signals maps ``flag_three_of_four`` onto ``flag_ror_prr_bcpnn``
        on read. Building the path locally skipped that mapping, so a pre-rename
        partition raised "lacks ['flag_ror_prr_bcpnn']" and was unreadable.
        """
        _write_signal(tmp_path, _legacy_frame([("LIPITOR 10MG", "X", True, False)]))
        _, n_total = scope.select(settings=_settings(tmp_path), overrides={}, rxcuis={})
        assert n_total == 1


class TestThePredicate:
    def test_only_flagged_pairs_count(self, tmp_path: Path) -> None:
        _write_signal(
            tmp_path,
            _frame(
                [
                    ("FLAGGED", "X", True, False),
                    ("FLAGGED", "Y", True, False),
                    ("UNFLAGGED", "X", False, False),
                ]
            ),
        )
        counts = scope.flagged_pair_counts(settings=_settings(tmp_path))
        assert counts.to_dicts() == [{"drug": "FLAGGED", "flagged_pairs": 2}]

    def test_insufficient_pairs_are_excluded_when_they_exist(self, tmp_path: Path) -> None:
        """The conjunct is a no-op on the measured runs but is not dead code."""
        _write_signal(
            tmp_path,
            _frame([("A", "X", True, True), ("A", "Y", True, False), ("B", "X", True, True)]),
        )
        counts = scope.flagged_pair_counts(settings=_settings(tmp_path))
        assert counts.to_dicts() == [{"drug": "A", "flagged_pairs": 1}]

    def test_flag_all_four_is_never_consulted(self, tmp_path: Path) -> None:
        """MGPS is provisional, so scoping on it would make the corpus unquotable."""
        frame = _frame([("A", "X", True, False), ("B", "X", False, False)]).with_columns(
            pl.Series("flag_all_four", [False, True])
        )
        _write_signal(tmp_path, frame)
        counts = scope.flagged_pair_counts(settings=_settings(tmp_path))
        assert counts["drug"].to_list() == ["A"]


class TestTheCap:
    @pytest.fixture
    def populated(self, tmp_path: Path) -> Settings:
        rows = []
        for name, pairs in (("HIGH", 5), ("MID", 3), ("LOW", 1)):
            rows.extend([(name, f"PT{index}", True, False) for index in range(pairs)])
        # Two strings with equal counts, to pin the tie-break.
        rows.extend([("BETA", "PT0", True, False), ("ALPHA", "PT0", True, False)])
        _write_signal(tmp_path, _frame(rows))
        return _settings(tmp_path)

    def test_n_total_is_the_whole_population_not_the_capped_slice(
        self, populated: Settings
    ) -> None:
        units, n_total = scope.select(top_k=2, settings=populated, overrides={}, rxcuis={})
        assert n_total == 5
        assert len(units) == 2

    def test_selection_is_by_descending_flagged_pairs(self, populated: Settings) -> None:
        """Third place goes to ALPHA, not LOW: three strings tie on one pair and
        the tie-break is the string, so LOW does not outrank ALPHA or BETA."""
        units, _ = scope.select(top_k=3, settings=populated, overrides={}, rxcuis={})
        assert [unit.folded_string for unit in units] == ["HIGH", "MID", "ALPHA"]
        assert [unit.flagged_pairs for unit in units] == [5, 3, 1]

    def test_ties_break_on_the_string_ascending(self, populated: Settings) -> None:
        units, _ = scope.select(top_k=5, settings=populated, overrides={}, rxcuis={})
        tied = [unit.folded_string for unit in units if unit.flagged_pairs == 1]
        assert tied == ["ALPHA", "BETA", "LOW"]

    def test_the_selection_is_deterministic_across_calls(self, populated: Settings) -> None:
        first, _ = scope.select(top_k=4, settings=populated, overrides={}, rxcuis={})
        second, _ = scope.select(top_k=4, settings=populated, overrides={}, rxcuis={})
        assert first == second

    def test_top_k_below_one_is_rejected(self, populated: Settings) -> None:
        with pytest.raises(SignalScopeError, match="at least 1"):
            scope.select(top_k=0, settings=populated, overrides={}, rxcuis={})


class TestRouting:
    @pytest.fixture
    def populated(self, tmp_path: Path) -> Settings:
        _write_signal(tmp_path, _frame([("LIPITOR 10MG", "X", True, False)]))
        return _settings(tmp_path)

    def test_an_override_wins_over_everything(self, populated: Settings) -> None:
        units, _ = scope.select(
            settings=populated,
            overrides={"LIPITOR 10MG": "ATORVASTATIN CALCIUM"},
            rxcuis={"LIPITOR 10MG": 83367},
        )
        assert units[0].route == "override"
        assert units[0].query == "ATORVASTATIN CALCIUM"
        assert units[0].ingredient_rxcui is None

    def test_the_ingredient_route_is_used_when_a_mapping_exists(self, populated: Settings) -> None:
        units, _ = scope.select(settings=populated, overrides={}, rxcuis={"LIPITOR 10MG": 83367})
        assert units[0].route == "ingredient"
        assert units[0].ingredient_rxcui == 83367
        assert units[0].manifest_unit == "rxcui:83367"

    def test_without_normalization_every_string_takes_the_cleaned_route(
        self, populated: Settings
    ) -> None:
        """The state this pipeline actually runs in: RxNorm is not being rebuilt."""
        units, _ = scope.select(settings=populated, overrides={}, rxcuis={})
        assert units[0].route == "cleaned_string"
        assert units[0].query == "LIPITOR"
        assert units[0].ingredient_rxcui is None

    def test_a_string_routed_unit_fits_the_manifest_column(self, populated: Settings) -> None:
        """IngestManifest.unit is 32 characters and a drug string is not."""
        units, _ = scope.select(settings=populated, overrides={}, rxcuis={})
        unit = units[0].manifest_unit
        assert unit.startswith("str:")
        assert len(unit) <= 32

    def test_the_manifest_unit_is_stable_for_the_same_query(self) -> None:
        made = _made("LIPITOR 10MG", "LIPITOR")
        assert made.manifest_unit == made.manifest_unit


def _made(folded: str, query: str, rxcui: int | None = None) -> ScopeUnit:
    return ScopeUnit(
        folded_string=folded,
        query=query,
        route="ingredient" if rxcui else "cleaned_string",
        ingredient_rxcui=rxcui,
        flagged_pairs=1,
    )


class TestTheManifestKeyIsTheQuery:
    """A cleaner change must invalidate the unit without anyone remembering to.

    Keyed on the folded string, the manifest asserted a unit was complete
    against a query the cleaner had stopped emitting: after the trailing period
    was folded, a re-run skipped all 200 units as already done.
    """

    def test_two_strings_that_fold_to_one_query_share_a_key(self) -> None:
        assert _made("GABAPENTIN.", "GABAPENTIN").manifest_unit == (
            _made("GABAPENTIN", "GABAPENTIN").manifest_unit
        )

    def test_a_three_way_group_shares_one_key(self) -> None:
        keys = {
            _made(name, "RITUXIMAB").manifest_unit
            for name in ("RITUXIMAB", "RITUXIMAB.", "RITUXIMAB (UNKNOWN)")
        }
        assert len(keys) == 1

    def test_changing_the_query_changes_the_key(self) -> None:
        """The whole point: the cleaner moves, the key moves with it."""
        before = _made("GABAPENTIN.", "GABAPENTIN.").manifest_unit
        after = _made("GABAPENTIN.", "GABAPENTIN").manifest_unit
        assert before != after

    def test_the_folded_string_alone_does_not_change_the_key(self) -> None:
        """The query is the work. Who asked for it is not part of the identity."""
        assert _made("A", "SHARED").manifest_unit == _made("B", "SHARED").manifest_unit

    def test_the_ingredient_route_keys_on_the_rxcui_not_the_query(self) -> None:
        """That route's identity is the concept; the query is derived from it."""
        assert _made("ANY", "IGNORED", rxcui=83367).manifest_unit == "rxcui:83367"

    def test_the_key_still_fits_the_manifest_column(self) -> None:
        key = _made("X" * 500, "Y" * 500).manifest_unit
        assert key.startswith("str:")
        assert len(key) <= 32


class TestTheQueryCleaner:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("LIPITOR 10MG", "LIPITOR"),
            ("ASPIRIN 81 MG TABLET", "ASPIRIN"),
            ("METFORMIN HCL 500MG ER", "METFORMIN HCL"),
            ("HUMIRA", "HUMIRA"),
            ("PREDNISONE 5 MG TABLETS", "PREDNISONE"),
            ("INSULIN GLARGINE 100 UNITS/ML SOLUTION", "INSULIN GLARGINE"),
            ("TYLENOL (ACETAMINOPHEN) 500MG", "TYLENOL"),
            ("AMOXICILLIN 875MG/CLAVULANATE 125MG", "AMOXICILLIN 875MG/CLAVULANATE"),
        ],
    )
    def test_trailing_dose_and_form_tokens_are_stripped(self, raw: str, expected: str) -> None:
        assert scope.clean_query(raw) == expected

    def test_interior_tokens_are_never_removed(self) -> None:
        """'MG' inside a name is part of the name, not a strength."""
        assert scope.clean_query("MG SULFATE INJECTION") == "MG SULFATE"

    def test_a_string_of_only_droppable_tokens_still_yields_a_query(self) -> None:
        assert scope.clean_query("10MG TABLET") == "10MG"

    def test_an_empty_string_yields_an_empty_query(self) -> None:
        assert scope.clean_query("   ") == ""

    def test_the_cleaner_is_deterministic(self) -> None:
        assert scope.clean_query("Lipitor 10mg") == scope.clean_query("LIPITOR 10MG")


class TestTheTrailingPeriodIsFolded:
    """openFDA folds it server-side; paying a request set to learn that is waste.

    Measured on the P05 run: 30 of the 200 selected strings were
    period-suffixed, all 30 had a bare twin in the same selection, and all 30
    pairs returned identical non-empty document sets.
    """

    def test_a_period_suffixed_string_and_its_bare_twin_produce_one_query(self) -> None:
        assert scope.clean_query("GABAPENTIN.") == scope.clean_query("GABAPENTIN")
        assert scope.clean_query("GABAPENTIN.") == "GABAPENTIN"

    @pytest.mark.parametrize(
        "raw", ["CIPROFLOXACIN.", "MYCOPHENOLATE MOFETIL.", "ALENDRONATE SODIUM."]
    )
    def test_the_fold_survives_the_dose_token_peel(self, raw: str) -> None:
        assert scope.clean_query(raw) == scope.clean_query(raw.rstrip("."))

    def test_a_run_of_periods_folds_the_same_way(self) -> None:
        """Nothing in the population needs this - no string ends in more than
        one period - but rstrip folds a run, and that is the behaviour openFDA
        would apply, so it is asserted rather than left to be discovered."""
        assert scope.clean_query("GABAPENTIN..") == "GABAPENTIN"

    def test_an_interior_period_is_never_touched(self) -> None:
        assert "." in scope.clean_query("VIT. B12 COMPLEX")

    def test_a_decimal_strength_is_not_damaged(self) -> None:
        """The strip is on the joined query, never on a token."""
        assert scope.clean_query("WARFARIN 0.5MG") == "WARFARIN"
        assert scope.clean_query("SOMETHING 2.5") == "SOMETHING"

    def test_a_string_of_only_periods_still_yields_a_query(self) -> None:
        """The non-empty invariant survives the fold: an empty query is
        recorded by the pipeline as a cleaner error, which this is not."""
        assert scope.clean_query(".") != ""


class TestTheOverrideFile:
    def test_an_absent_file_is_empty_and_not_an_error(self, tmp_path: Path) -> None:
        assert scope.load_overrides(tmp_path / "nothing.csv") == {}

    def test_rows_are_read_and_folded(self, tmp_path: Path) -> None:
        path = tmp_path / "spl_query.csv"
        path.write_text(
            "folded_string,openfda_query\nlipitor 10mg,ATORVASTATIN\n", encoding="utf-8"
        )
        assert scope.load_overrides(path) == {"LIPITOR 10MG": "ATORVASTATIN"}

    def test_incomplete_rows_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "spl_query.csv"
        path.write_text(
            "folded_string,openfda_query\nA,\n,B\nC,D\n",
            encoding="utf-8",
        )
        assert scope.load_overrides(path) == {"C": "D"}

    def test_the_committed_file_is_header_only(self) -> None:
        """It is Hassan's to curate. Nothing in this project writes its rows."""
        assert scope.load_overrides() == {}
