"""The signal-carrying scope: its predicate, its cap, and the query cleaner."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from signaldesk.core.config import Settings
from signaldesk.ingest.spl import scope
from signaldesk.ingest.spl.scope import ScopeUnit, SignalScopeError

pytestmark = pytest.mark.unit


def _settings(tmp_path: Path) -> Settings:
    return Settings(django_secret_key="test", data_dir=tmp_path, cache_dir=tmp_path / "cache")


def _write_signal(tmp_path: Path, frame: pl.DataFrame, run_id: str = "20260101T000000Z") -> None:
    partition = tmp_path / "parquet" / "signal" / f"run_id={run_id}"
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
        with pytest.raises(SignalScopeError, match="no run partitions"):
            scope.select(settings=_settings(tmp_path))

    def test_a_named_run_that_does_not_exist_is_an_error(self, tmp_path: Path) -> None:
        _write_signal(tmp_path, _frame([("A", "X", True, False)]))
        with pytest.raises(SignalScopeError, match="no parquet"):
            scope.select("20990101T000000Z", settings=_settings(tmp_path))

    def test_a_frame_missing_the_predicate_columns_is_an_error(self, tmp_path: Path) -> None:
        _write_signal(tmp_path, pl.DataFrame({"drug": ["A"], "pt": ["X"]}))
        with pytest.raises(SignalScopeError, match="lacks"):
            scope.select(settings=_settings(tmp_path))


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

    def test_the_manifest_unit_is_stable_for_the_same_string(self) -> None:
        made = ScopeUnit(
            folded_string="LIPITOR 10MG",
            query="LIPITOR",
            route="cleaned_string",
            ingredient_rxcui=None,
            flagged_pairs=1,
        )
        assert made.manifest_unit == made.manifest_unit


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
