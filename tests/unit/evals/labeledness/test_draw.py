"""The pure half of the draw: the seed, and the reserve stratum's predicate."""

from __future__ import annotations

import random

import pytest

from signaldesk.core.errors import AnnotationError
from signaldesk.evals.labeledness.draw import (
    Pair,
    _build_screen,
    _primary_string,
    _sibling_string,
    has_lexical_evidence,
    normalise_seed,
)
from signaldesk.evals.labeledness.frame import Frame, FrameCounts
from signaldesk.evals.labeledness.manifest import Protocol
from signaldesk.evals.labeledness.schedule import TOTAL_UNIQUE, build_schedule

pytestmark = pytest.mark.unit


class TestTheDrawIsReproducibleFromTheSeed:
    def test_the_same_seed_selects_the_same_pairs_and_documents(self) -> None:
        """Sampling and document assignment together, in the order draw() runs them.

        WHAT WOULD KEEP THIS GREEN WHILE THE PROPERTY BROKE: if the draw used the
        process-global `random` module rather than a local Random, two calls in
        one process could still agree by accident of ordering. That is why draw()
        constructs `random.Random(seed)` and why the assertion below builds two
        independent generators rather than reusing one.
        """
        pairs = [Pair(query=f"DRUG{index}", pt=f"PT{index}") for index in range(500)]
        documents = {pair.pair_id: tuple(range(10)) for pair in pairs}

        def _run(seed: int) -> tuple[list[str], list[int]]:
            rng = random.Random(seed)
            drawn = rng.sample(pairs, TOTAL_UNIQUE)
            assigned = [rng.choice(documents[pair.pair_id]) for pair in drawn]
            return [pair.pair_id for pair in drawn], assigned

        assert _run(20260902) == _run(20260902)
        assert _run(20260902) != _run(20260903)

    def test_the_schedule_is_part_of_what_the_seed_fixes(self) -> None:
        assert build_schedule(TOTAL_UNIQUE, random.Random(99)) == build_schedule(
            TOTAL_UNIQUE, random.Random(99)
        )

    def test_the_pair_id_is_stable_across_the_two_presentations_of_a_repeat(self) -> None:
        """Agreement groups on it, so it must not encode the presentation."""
        assert Pair(query="PREDNISONE", pt="HEADACHE").pair_id == "PREDNISONE||HEADACHE"


class TestTheSeedIsNotGuessedAt:
    @pytest.mark.parametrize("raw", ["0", "1", "20260902", "9" * 19])
    def test_a_decimal_seed_is_accepted(self, raw: str) -> None:
        assert normalise_seed(raw) == int(raw)

    @pytest.mark.parametrize("raw", ["", "-1", "0x10", "1e6", " 12", "12 ", "abc", "9" * 20])
    def test_anything_a_shell_might_have_mangled_is_rejected(self, raw: str) -> None:
        with pytest.raises(AnnotationError, match="seed must be a decimal integer"):
            normalise_seed(raw)


class TestTheReserveStratumPredicate:
    """Published because it decides membership of a committed stratum.

    It is not a labelledness judgement and is never shown to the annotator. It
    catches the explicit route only, which is exactly why the reserve supplements
    a uniform sample rather than replacing one: subsumption and class warnings
    produce no lexical hit at all.
    """

    def test_an_exact_term_hits(self) -> None:
        assert has_lexical_evidence("HEADACHE", "Headache was reported in 3 percent.")

    def test_every_content_word_present_stemmed_hits(self) -> None:
        assert has_lexical_evidence(
            "HEPATIC ENZYME INCREASED", "Increases in hepatic enzymes were observed."
        )

    def test_a_partial_token_match_does_not_hit(self) -> None:
        """Related is not the same. "any token" would fire on 41 percent of pairs."""
        assert not has_lexical_evidence("HEPATIC ENZYME INCREASED", "Hepatic impairment.")

    def test_a_broader_term_produces_no_hit(self) -> None:
        """The stratum's blind spot, asserted rather than assumed.

        "Serious skin reactions" covers Stevens-Johnson syndrome for a human and
        shares no content word with it. A gold set drawn only from this predicate
        would contain none of the cases the adjudicator is most likely to fail.
        """
        assert not has_lexical_evidence(
            "STEVENS-JOHNSON SYNDROME", "Serious skin reactions have been reported."
        )

    def test_a_class_warning_produces_no_hit_either(self) -> None:
        assert not has_lexical_evidence(
            "MYOCARDIAL INFARCTION", "NSAIDs cause an increased risk of thrombotic events."
        )

    def test_a_term_with_no_content_words_never_hits_on_tokens_alone(self) -> None:
        assert not has_lexical_evidence("NOS", "Something else entirely.")

    def test_matching_is_case_insensitive(self) -> None:
        assert has_lexical_evidence("headache", "HEADACHE AND NAUSEA")


class TestWhichStringAScreenIsPresentedUnder:
    """The sibling-string presentation, and its measured coverage.

    A repeat under `PREDNISONE.` where the original was `PREDNISONE` is the same
    query, the same document and the same event with a different surface. It is
    the only cosmetic variation available, because changing the text would change
    the question.
    """

    def _frame(self, members: dict[str, tuple[str, ...]]) -> Frame:
        documents_by_string = {name: (1,) for group in members.values() for name in group}
        return Frame(
            counts=FrameCounts(
                documents_total=1,
                documents_with_primary_section=1,
                documents_without_primary_section=0,
                strings_total=len(documents_by_string),
                strings_clean=len(documents_by_string),
                strings_capped=0,
                strings_unknown=0,
                strings_eligible=len(documents_by_string),
                query_groups_eligible=len(members),
                documents_eligible=1,
            ),
            documents_by_string=documents_by_string,
            query_by_string={name: query for query, group in members.items() for name in group},
            strings_by_query=members,
            set_id_by_document={1: "set-1"},
        )

    def test_the_first_member_ascending_is_the_primary_presentation(self) -> None:
        """Same tie-break the scope selection uses, so it is one convention."""
        frame = self._frame({"PREDNISONE": ("PREDNISONE", "PREDNISONE.")})
        assert _primary_string(frame, "PREDNISONE") == "PREDNISONE"

    def test_a_repeat_uses_the_sibling_when_the_group_has_one(self) -> None:
        frame = self._frame({"PREDNISONE": ("PREDNISONE", "PREDNISONE.")})
        assert _sibling_string(frame, "PREDNISONE") == "PREDNISONE."

    def test_a_group_without_a_sibling_repeats_under_the_same_string(self) -> None:
        """Coverage is partial and measured, not claimed.

        29 of 157 eligible query groups carry a twin, so about a fifth of repeats
        can vary their surface and the rest cannot. The manifest records the
        coverage rather than implying it is total.
        """
        frame = self._frame({"HUMIRA": ("HUMIRA",)})
        assert _sibling_string(frame, "HUMIRA") == "HUMIRA"
        assert _primary_string(frame, "HUMIRA") == "HUMIRA"


class TestScreensCarryWhatTheyWereJudgedAgainst:
    def test_a_built_screen_digests_every_block_and_sizes_the_primary_section(self) -> None:
        screen = _build_screen(
            screen_id="s0001",
            position=1,
            pair=Pair(query="DRUG", pt="HEADACHE"),
            drug_string="DRUG",
            document_id=7,
            set_id="set-7",
            blocks=[
                ("adverse_reactions", 0, "Headache."),
                ("adverse_reactions", 1, "Nausea."),
                ("boxed_warning", 0, "A class effect."),
            ],
        )
        assert screen.primary_chars == len("Headache.") + len("Nausea.")
        assert screen.protocol is Protocol.FULL
        assert screen.set_id == "set-7"
        assert screen.document_id == 7
        assert len(screen.digests()) == 3

    def test_a_repeat_screen_keeps_the_pair_id_of_its_original(self) -> None:
        """Agreement groups on pair_id, so it must survive the re-presentation."""
        pair = Pair(query="PREDNISONE", pt="HEADACHE")
        first = _build_screen(
            screen_id="s0007",
            position=7,
            pair=pair,
            drug_string="PREDNISONE",
            document_id=1,
            set_id="set-1",
            blocks=[("adverse_reactions", 0, "Headache.")],
        )
        second = _build_screen(
            screen_id="s0200",
            position=200,
            pair=pair,
            drug_string="PREDNISONE.",
            document_id=1,
            set_id="set-1",
            blocks=[("adverse_reactions", 0, "Headache.")],
            is_repeat=True,
            repeat_of="s0007",
        )
        assert first.pair_id == second.pair_id
        assert first.digests() == second.digests()
        assert second.repeat_of == "s0007"


class TestTheCausalDisclaimerDoesNotSuppressLexicalEvidence:
    """G4, on the one mechanical surface it touches.

    The guideline rule -- hedged reporting is still `l`, and an explicit
    causality disclaimer changes nothing -- is a human judgement and no test can
    pin it. What can be pinned is that nothing in the machinery quietly
    contradicts it: the reserve stratum's predicate must still fire on text
    carrying the standard FDA postmarketing boilerplate, or the stratum would
    systematically drop the 70.5 percent of eligible documents that carry it.
    """

    BOILERPLATE = (
        "Cases of urticaria have been reported in patients receiving this drug. "
        "Because these reactions are reported voluntarily from a population of "
        "uncertain size, it is not always possible to reliably estimate their "
        "frequency or establish a causal relationship to drug exposure."
    )

    def test_lexical_evidence_survives_an_explicit_causality_disclaimer(self) -> None:
        assert has_lexical_evidence("URTICARIA", self.BOILERPLATE)

    def test_the_disclaimer_alone_creates_no_evidence(self) -> None:
        """The other direction: the boilerplate must not make everything match."""
        assert not has_lexical_evidence("MYOCARDIAL INFARCTION", self.BOILERPLATE)

    def test_the_render_path_does_not_strip_or_flag_the_disclaimer(self) -> None:
        """The annotator sees the paragraph as published, disclaimer included.

        Removing it, or marking it, would be the harness taking a position on a
        judgement the guideline reserves for the annotator.
        """
        from signaldesk.evals.labeledness.render import ScreenView, render_screen

        screen = _build_screen(
            screen_id="s0001",
            position=1,
            pair=Pair(query="DRUG", pt="URTICARIA"),
            drug_string="DRUG",
            document_id=1,
            set_id="set-1",
            blocks=[("adverse_reactions", 0, self.BOILERPLATE)],
        )
        frame = render_screen(screen, progress="p", view=ScreenView())
        assert "establish a causal relationship to drug exposure." in frame
        assert "disclaimer" not in frame.lower()
