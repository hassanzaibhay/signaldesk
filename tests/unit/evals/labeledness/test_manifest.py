"""The committed manifest: ASCII, digest-verified, and self-consistent."""

from __future__ import annotations

from pathlib import Path

import pytest
from _builders import build_manifest, build_screen

from signaldesk.core.errors import AnnotationError
from signaldesk.evals.labeledness.manifest import (
    SectionBlock,
    digest,
    manifest_filename,
    read_manifest,
    write_manifest,
)

pytestmark = pytest.mark.unit

#: Label text really does carry these: micro signs in dose tables, degree signs
#: in storage instructions, and the occasional typographic dash. Written as
#: escapes because this file is tracked and tracked files are ASCII. What the
#: string holds at runtime is the real thing, which is what the assertions need.
NON_ASCII_LABEL_TEXT = "Reactions at 5 \u00b5g. Store below 25 \u00b0C. Rate was 1\u20132 percent."


class TestTheArtifactIsAscii:
    def test_non_ascii_label_text_survives_as_escapes(self, tmp_path: Path) -> None:
        """ensure_ascii is what lets this sit under evals/history/ unexempted.

        evals/history/ is not on the hygiene ASCII allowlist, so the manifest has
        to be pure ASCII on disk while still carrying label text verbatim.
        Escaping costs about half a percent on this corpus.
        """
        assert not NON_ASCII_LABEL_TEXT.isascii()
        manifest = build_manifest(
            [build_screen(blocks=[("adverse_reactions", 0, NON_ASCII_LABEL_TEXT)])]
        )
        path = tmp_path / "sample.json"
        write_manifest(manifest, path)

        raw = path.read_bytes()
        assert all(byte < 0x80 for byte in raw)
        assert rb"\u00b5" in raw

        reloaded = read_manifest(path)
        assert reloaded.screens[0].sections[0].text == NON_ASCII_LABEL_TEXT

    def test_the_filename_carries_the_run_id(self) -> None:
        assert manifest_filename("20260902T101112Z") == "labeledness_sample_20260902T101112Z.json"


class TestDigestsPinTheText:
    def test_a_block_digest_is_the_hash_of_its_own_text(self) -> None:
        block = SectionBlock.build(section_code="adverse_reactions", ordinal=0, text="Headache.")
        assert block.sha256 == digest("Headache.")

    def test_a_manifest_edited_after_writing_fails_to_load(self, tmp_path: Path) -> None:
        """Validation is the mechanism, not a formality.

        Every digest is re-derived on load, so a manifest whose text was changed
        after it was committed stops the run rather than scoring against text
        nobody saw.
        """
        path = tmp_path / "sample.json"
        write_manifest(build_manifest([build_screen()]), path)
        tampered = path.read_text(encoding="ascii").replace(
            "Nausea and vomiting were reported.", "Nausea and vomiting were NOT reported."
        )
        path.write_text(tampered, encoding="ascii")

        with pytest.raises(AnnotationError, match="malformed"):
            read_manifest(path)

    def test_the_digest_map_is_keyed_by_section_and_ordinal(self) -> None:
        screen = build_screen(
            blocks=[
                ("adverse_reactions", 0, "First block."),
                ("adverse_reactions", 1, "Second block."),
                ("boxed_warning", 0, "A class effect."),
            ]
        )
        assert set(screen.digests()) == {
            "adverse_reactions[0]",
            "adverse_reactions[1]",
            "boxed_warning[0]",
        }
        assert screen.section_codes == ("adverse_reactions", "boxed_warning")


class TestTheScreenListIsWellFormed:
    def test_positions_must_be_dense_and_ordered(self) -> None:
        with pytest.raises(ValueError, match="ordered by position"):
            build_manifest(
                [
                    build_screen(screen_id="s0001", position=1),
                    build_screen(screen_id="s0003", position=3),
                ]
            )

    def test_screen_ids_must_be_unique(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            build_manifest(
                [
                    build_screen(screen_id="s0001", position=1),
                    build_screen(screen_id="s0001", position=2),
                ]
            )

    def test_screen_at_is_one_indexed(self) -> None:
        manifest = build_manifest(
            [
                build_screen(screen_id="s0001", position=1),
                build_screen(screen_id="s0002", position=2),
            ]
        )
        assert manifest.screen_at(1).screen_id == "s0001"
        assert manifest.screen_at(2).screen_id == "s0002"


class TestProvenanceIsCarried:
    def test_the_manifest_joins_the_signal_run_to_the_label_artifact(self) -> None:
        """The link the SPL ingest artifact does not record.

        The label ingest artifact says nothing about which signal run its scope
        came from. Recording both here, with the partition count that was
        actually on disk, turns that from an unstated premise into a checked
        fact.
        """
        manifest = build_manifest([build_screen()])
        assert manifest.signal_run_id == "20260831T090758Z"
        assert manifest.signal_artifact.startswith("signals_")
        assert manifest.spl_artifact.startswith("spl_ingest_")
        assert manifest.signal_partitions_on_disk == 1

    def test_the_frame_counts_are_committed_not_recomputed(self) -> None:
        manifest = build_manifest([build_screen()])
        assert manifest.frame.documents_eligible == 8878
        assert manifest.frame.strings_capped == 3
        assert manifest.frame.documents_without_primary_section == 4040
        assert manifest.frame_pairs == 374846
