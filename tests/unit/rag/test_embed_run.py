"""What an embed run records about the hardware it ran on.

The device fields reach the artifact by riding the embedding dict into
``dense``: ``artifact.collect`` already spreads it. So the thing worth testing is
the end of that chain, from a torch state through resolution to the dict the
artifact is built from, and in particular what it declines to carry.

Nothing here touches a database. ``EmbedRun`` is a dataclass and ``as_dict`` is
pure.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest
from faketorch import cpu_only_wheel, usable_gpu

from signaldesk.rag import device
from signaldesk.rag.index import artifact
from signaldesk.rag.index.corpus import EMBED_BATCH_ROWS, EmbedRun, embed_pending

pytestmark = pytest.mark.unit


def make_run(choice: device.DeviceChoice | None) -> EmbedRun:
    began = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
    return EmbedRun(
        model="ncbi/MedCPT-Article-Encoder",
        model_revision="d05a736da4bb84ee4057b7f7999485be6ed85465",
        embedded_this_run=256,
        already_embedded=0,
        remaining=0,
        batches=1,
        seconds=100.0,
        started_at=began,
        finished_at=began,
        rows_first_written_in_run=256,
        device=choice,
    )


class TestWhatTheRunRecordsAboutItsHardware:
    def test_a_cpu_run_on_a_gpu_machine_records_no_gpu(self) -> None:
        """The strongest of these. An available card that was not used must not
        appear in the record, or a later reader concludes the vectors came off it.
        """
        choice = device.resolve(device.CPU, torch_module=usable_gpu())

        recorded = make_run(choice).as_dict()

        assert recorded["device"] == device.CPU
        assert recorded["gpu_name"] is None
        assert recorded["cuda_version"] is None

    def test_a_cuda_run_records_what_torch_reported(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=usable_gpu())

        recorded = make_run(choice).as_dict()

        assert recorded["device"] == device.CUDA
        assert recorded["cuda_version"] == "12.4"
        assert recorded["precision"] == device.FP32

    def test_the_cpu_only_container_records_its_wheel(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=cpu_only_wheel())

        recorded = make_run(choice).as_dict()

        assert recorded["device"] == device.CPU
        assert recorded["torch_version"] == "2.13.0+cpu"

    def test_a_run_with_no_resolved_device_records_none_of_these_fields(self) -> None:
        """Absent rather than guessed. A run that never resolved a device has no
        device to report, and a defaulted "cpu" would be a claim nothing made.
        """
        recorded = make_run(None)

        assert "device" not in recorded.as_dict()
        assert "torch_version" not in recorded.as_dict()

    def test_the_run_still_reports_everything_it_did_before(self) -> None:
        """The device fields are additive; nothing the artifact already quotes moves."""
        recorded = make_run(device.resolve(device.AUTO, torch_module=cpu_only_wheel()))

        for key in (
            "ran",
            "model",
            "model_revision",
            "chunks_embedded_this_run",
            "chunks_already_embedded",
            "chunks_without_embedding",
            "batches",
            "seconds",
            "chunks_per_second",
            "rows_first_written_in_run",
            "run_window",
        ):
            assert key in recorded.as_dict(), key


class TestTheWriteBatchIsUnchanged:
    """Regression pin. No failing-first demonstration: this passes on HEAD.

    It is here to protect a structure, not to demonstrate a fix. The committed
    record index_20260907T070358Z.json rests on writer_evidence finding 244
    transactions of 256 rows, and that evidence is what names the record for a
    single run. A later edit that made the database write batch follow the encode
    batch would invalidate the reasoning behind an artifact already published,
    and nothing else in the suite would notice.
    """

    def test_the_database_write_batch_is_still_256(self) -> None:
        assert EMBED_BATCH_ROWS == 256

    def test_writer_evidence_still_reads_the_table_in_256s(self) -> None:
        default = inspect.signature(artifact.writer_evidence).parameters["batch_rows"].default

        assert default == 256

    def test_the_encode_batch_cannot_reach_the_write_batch(self) -> None:
        """They are separate parameters, and a device default moves only one."""
        parameters = inspect.signature(embed_pending).parameters

        assert parameters["batch_rows"].default == EMBED_BATCH_ROWS
        assert parameters["batch_size"].default != parameters["batch_rows"].default
