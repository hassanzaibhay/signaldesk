"""What a screen shows, and everything it must not."""

from __future__ import annotations

import dataclasses

import pytest
from _builders import build_screen

from signaldesk.evals.labeledness.manifest import LONG_SECTION_CHARS, Protocol, protocol_for
from signaldesk.evals.labeledness.render import (
    ScreenView,
    content_words,
    find_matches,
    render_screen,
    search_terms,
    stem,
)

pytestmark = pytest.mark.unit


class TestARepeatIsIndistinguishable:
    def test_the_frame_does_not_depend_on_repeat_of(self) -> None:
        """The marker must not leak into anything rendered.

        Asserted by clearing the field and comparing bytes, rather than by
        grepping the output for a word like "repeat". A leak through some other
        phrasing would pass a grep and fails this.

        Scope, stated because it is the blind spot: this covers the rendered
        frame only. A marker escaping through a window title, a log line or a
        terminal bell would not move these bytes. The session test that no log
        line names the field is what covers the log; nothing covers a title,
        because nothing sets one.
        """
        repeat = build_screen(screen_id="s0200", position=200, is_repeat=True, repeat_of="s0007")
        cleared = repeat.model_copy(update={"is_repeat": False, "repeat_of": None})

        rendered_repeat = render_screen(repeat, progress="screen 200 of 330", view=ScreenView())
        rendered_plain = render_screen(cleared, progress="screen 200 of 330", view=ScreenView())
        assert rendered_repeat == rendered_plain

    def test_a_sibling_presentation_differs_only_in_the_drug_string(self) -> None:
        """The one deliberate difference between a repeat and its original."""
        original = build_screen(screen_id="s0007", position=7, drug_string="PREDNISONE")
        sibling = build_screen(
            screen_id="s0200",
            position=200,
            drug_string="PREDNISONE.",
            is_repeat=True,
            repeat_of="s0007",
        )
        progress = "screen 7 of 330"
        first = render_screen(original, progress=progress, view=ScreenView())
        second = render_screen(sibling, progress=progress, view=ScreenView())
        assert first != second
        assert first.replace("PREDNISONE", "X") == second.replace("PREDNISONE.", "X").replace(
            "PREDNISONE", "X"
        )


class TestNothingCorrelatedWithTheAnswerIsShown:
    def test_no_statistic_stratum_or_tally_reaches_the_frame(self) -> None:
        screen = build_screen()
        frame = render_screen(screen, progress="screen 1 of 330", view=ScreenView()).lower()
        for forbidden in (
            "ror",
            "prr",
            "ic025",
            "ebgm",
            "flag",
            "stratum",
            "repeat",
            "labelled so far",
            "pair_id",
        ):
            assert forbidden not in frame, forbidden

    def test_the_progress_line_carries_position_and_nothing_else(self) -> None:
        screen = build_screen()
        frame = render_screen(screen, progress="screen 12 of 330", view=ScreenView())
        assert "screen 12 of 330" in frame

    def test_the_document_identity_is_shown_so_a_verdict_is_attributable(self) -> None:
        screen = build_screen(set_id="abc-123")
        assert "abc-123" in render_screen(screen, progress="p", view=ScreenView())


class TestSearchIsOfferedAndNotApplied:
    def test_the_required_terms_are_listed_but_no_text_is_pre_highlighted(self) -> None:
        """Fewer than 3 percent of pairs have the term verbatim in the text.

        A pre-highlight would turn its own absence into evidence of absence. The
        terms are printed so the protocol is in front of the annotator; running
        them is a keypress.
        """
        screen = build_screen(
            pt="HEPATIC ENZYME INCREASED",
            blocks=[("adverse_reactions", 0, "Hepatic enzyme increased was observed.")],
        )
        frame = render_screen(screen, progress="p", view=ScreenView())
        assert "hepatic" in frame
        assert "/  search" in frame
        # The body is rendered verbatim, with no marker inserted around a hit.
        assert "Hepatic enzyme increased was observed." in frame

    def test_the_next_and_previous_match_keys_are_not_verdict_keys(self) -> None:
        """n is a verdict. Overloading it onto navigation writes a label on a miskey."""
        screen = build_screen()
        frame = render_screen(screen, progress="p", view=ScreenView(query="nausea"))
        assert ". next" in frame
        assert ", prev" in frame

    @pytest.mark.parametrize(
        ("pt", "expected"),
        [
            ("HEADACHE", ("headache",)),
            ("HEPATIC ENZYME INCREASED", ("hepatic", "enzyme", "increas")),
            ("RASH NOS", ("rash",)),
            ("SKIN DISORDER", ("skin",)),
        ],
    )
    def test_content_words_drop_filler_and_short_words(
        self, pt: str, expected: tuple[str, ...]
    ) -> None:
        assert tuple(stem(word) for word in content_words(pt)) == expected

    def test_search_terms_lead_with_the_verbatim_term(self) -> None:
        terms = search_terms("HEPATIC ENZYME INCREASED")
        assert terms[0] == "hepatic enzyme increased"
        assert "increas" in terms

    def test_find_matches_is_case_insensitive_and_finds_every_offset(self) -> None:
        assert find_matches("Rash and rash and RASH", "rash") == (0, 9, 18)
        assert find_matches("nothing here", "rash") == ()


class TestTheReadingProtocol:
    def test_the_threshold_puts_long_sections_on_the_bounded_protocol(self) -> None:
        assert protocol_for(LONG_SECTION_CHARS) is Protocol.FULL
        assert protocol_for(LONG_SECTION_CHARS + 1) is Protocol.BOUNDED

    def test_the_bounded_protocol_is_announced_on_the_screen(self) -> None:
        """The annotator has to know an n here is a defensible n, not a shortcut."""
        long_text = "word " * 4000
        screen = build_screen(blocks=[("adverse_reactions", 0, long_text)])
        assert screen.protocol is Protocol.BOUNDED
        frame = render_screen(screen, progress="p", view=ScreenView())
        assert "BOUNDED" in frame
        assert "defensible n" in frame

    def test_the_full_protocol_is_announced_too(self) -> None:
        frame = render_screen(build_screen(), progress="p", view=ScreenView())
        assert "FULL" in frame


class TestPagingAndSections:
    def test_only_present_sections_are_offered(self) -> None:
        screen = build_screen(
            blocks=[
                ("adverse_reactions", 0, "Reactions."),
                ("boxed_warning", 0, "A class effect."),
            ]
        )
        frame = render_screen(screen, progress="p", view=ScreenView())
        assert "(warnings)" in frame
        assert "boxed_warning" in frame

    def test_a_page_step_moves_the_window(self) -> None:
        text = "\n".join(f"line {index}" for index in range(200))
        screen = build_screen(blocks=[("adverse_reactions", 0, text)])
        first = render_screen(screen, progress="p", view=ScreenView())
        second = render_screen(screen, progress="p", view=ScreenView(page=1))
        assert first != second
        assert "page 1/" in first
        assert "page 2/" in second

    def test_a_search_jumps_the_page_to_the_match(self) -> None:
        text = "\n".join(f"line {index}" for index in range(200)) + "\nneedle here"
        screen = build_screen(blocks=[("adverse_reactions", 0, text)])
        jumped = render_screen(screen, progress="p", view=ScreenView(query="needle"))
        assert "needle here" in jumped

    def test_view_transitions_reset_what_they_should(self) -> None:
        view = ScreenView(section_index=0, page=4, query="x", match_index=2)
        assert dataclasses.replace(view).page == 4
        assert view.with_section(1).page == 0
        assert view.with_query("y").match_index == 0
        assert view.with_page(-5).page == 0
