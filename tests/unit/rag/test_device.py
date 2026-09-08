"""Device resolution, which decides where a model runs and records what it did.

Nothing here imports torch. Continuous integration installs the dev group and
never the ``ml`` extra, so torch does not exist in the job that runs this file
and there is nothing to monkeypatch. Every function under test takes the torch
module as an argument instead, and ``faketorch`` supplies it: the same
substitution monkeypatching would make, and the only one that runs in CI.

No test here is skipped for want of a GPU, because no test here needs one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from faketorch import cpu_only_wheel, unusable_gpu, usable_gpu

from signaldesk.rag import device
from signaldesk.rag.device import DeviceError

pytestmark = pytest.mark.unit


class TestTheThreeStates:
    """A CPU fallback has to say which of the three situations it is in."""

    def test_a_cpu_only_wheel_resolves_to_cpu_and_names_the_wheel(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=cpu_only_wheel())

        assert choice.name == device.CPU
        assert choice.reason == device.REASON_NO_CUDA_BUILD

    def test_a_cuda_build_that_cannot_reach_a_device_is_a_different_reason(self) -> None:
        """Not the same problem as a CPU-only wheel, and not reported as one."""
        choice = device.resolve(device.AUTO, torch_module=unusable_gpu())

        assert choice.name == device.CPU
        assert choice.reason == device.REASON_UNAVAILABLE
        assert choice.reason != device.REASON_NO_CUDA_BUILD

    def test_an_available_device_resolves_to_cuda_and_carries_its_identity(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=usable_gpu())

        assert choice.name == device.CUDA
        assert choice.reason == device.REASON_CUDA_AVAILABLE
        assert choice.gpu_name == "NVIDIA GeForce RTX 3050 Ti Laptop GPU"
        assert choice.cuda_version == "12.4"


class TestExplicitRequests:
    def test_asking_for_cuda_without_a_cuda_build_fails(self) -> None:
        with pytest.raises(DeviceError) as raised:
            device.resolve(device.CUDA, torch_module=cpu_only_wheel())

        assert device.REASON_NO_CUDA_BUILD in str(raised.value)

    def test_asking_for_cuda_when_it_is_unreachable_fails(self) -> None:
        """A request that cannot be honoured is an error, never a quiet fallback."""
        with pytest.raises(DeviceError) as raised:
            device.resolve(device.CUDA, torch_module=unusable_gpu())

        assert device.REASON_UNAVAILABLE in str(raised.value)

    def test_asking_for_cpu_on_a_gpu_machine_is_honoured(self) -> None:
        choice = device.resolve(device.CPU, torch_module=usable_gpu())

        assert choice.name == device.CPU
        assert choice.reason == device.REASON_CPU_REQUESTED

    def test_an_unknown_device_name_is_refused(self) -> None:
        with pytest.raises(DeviceError):
            device.resolve("mps", torch_module=usable_gpu())


class TestDeviceNodes:
    """Torch alone cannot tell "no card" from "card this wheel cannot use"."""

    def test_nodes_are_seen_when_they_are_there(self, tmp_path: Path) -> None:
        (tmp_path / "nvidia0").touch()

        assert device.gpu_device_nodes_present(tmp_path) is True

    def test_the_wsl_compute_node_counts(self, tmp_path: Path) -> None:
        (tmp_path / "dxg").touch()

        assert device.gpu_device_nodes_present(tmp_path) is True

    def test_an_empty_dev_has_none(self, tmp_path: Path) -> None:
        assert device.gpu_device_nodes_present(tmp_path) is False

    def test_a_missing_dev_has_none(self, tmp_path: Path) -> None:
        assert device.gpu_device_nodes_present(tmp_path / "nothing-here") is False

    def test_a_refusal_says_whether_a_device_node_was_visible(self, tmp_path: Path) -> None:
        """The message has to separate the two causes, not just name one."""
        with pytest.raises(DeviceError) as raised:
            device.resolve(device.CUDA, torch_module=cpu_only_wheel(), dev_root=tmp_path)

        assert "no gpu device node" in str(raised.value).lower()


class TestPrecision:
    def test_fp32_is_the_default(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=usable_gpu())

        assert choice.precision == device.FP32

    def test_fp16_is_recorded_when_asked_for_on_a_device_that_can_do_it(self) -> None:
        choice = device.resolve(device.AUTO, fp16=True, torch_module=usable_gpu())

        assert choice.precision == device.FP16

    def test_fp16_on_cpu_fails_rather_than_being_ignored(self) -> None:
        with pytest.raises(DeviceError):
            device.resolve(device.AUTO, fp16=True, torch_module=cpu_only_wheel())

    def test_fp32_runs_in_no_autocast_context_at_all(self) -> None:
        """The fp32 forward pass must be inert, not autocast-with-enabled-false."""
        torch = usable_gpu()
        choice = device.resolve(device.AUTO, torch_module=torch)

        with device.autocast_context(torch, choice):
            pass

        assert torch.autocast_calls == []

    def test_fp16_builds_an_autocast_for_the_resolved_device(self) -> None:
        torch = usable_gpu()
        choice = device.resolve(device.AUTO, fp16=True, torch_module=torch)

        device.autocast_context(torch, choice)

        assert torch.autocast_calls == [{"device_type": device.CUDA, "dtype": torch.float16}]


class TestOutOfMemory:
    """An OOM fails with the batch size that caused it. No retry, no fallback."""

    @staticmethod
    def _torch_oom() -> type[Exception]:
        kind = type("OutOfMemoryError", (RuntimeError,), {})
        kind.__module__ = "torch.cuda"
        return kind

    def test_a_torch_oom_becomes_a_device_error_naming_the_batch_size(self) -> None:
        with (
            pytest.raises(DeviceError) as raised,
            device.surfacing_oom(batch_size=64, device=device.CUDA),
        ):
            raise self._torch_oom()("CUDA out of memory")

        assert "64" in str(raised.value)
        assert device.CUDA in str(raised.value)

    def test_the_original_error_is_kept_as_the_cause(self) -> None:
        with (
            pytest.raises(DeviceError) as raised,
            device.surfacing_oom(batch_size=32, device=device.CUDA),
        ):
            raise self._torch_oom()("CUDA out of memory")

        assert type(raised.value.__cause__).__name__ == "OutOfMemoryError"

    def test_a_same_named_error_from_another_library_passes_through(self) -> None:
        """Matching on the name alone would swallow an unrelated failure."""
        impostor = type("OutOfMemoryError", (RuntimeError,), {})
        impostor.__module__ = "some_other_library"

        with (
            pytest.raises(impostor),
            device.surfacing_oom(batch_size=32, device=device.CUDA),
        ):
            raise impostor("not torch's")

    def test_an_unrelated_error_passes_through(self) -> None:
        with (
            pytest.raises(ValueError),
            device.surfacing_oom(batch_size=32, device=device.CUDA),
        ):
            raise ValueError("something else went wrong")

    def test_the_body_runs_exactly_once_and_nothing_is_raised(self) -> None:
        """A guard that yielded twice, or not at all, would be caught here.

        The count is derived rather than echoed back: asserting on a string the
        test itself put inside the block would prove only that Python executes
        statements.
        """
        runs = 0

        with device.surfacing_oom(batch_size=32, device=device.CPU):
            runs += 1

        assert runs == 1


class TestTheEncodeBatchSize:
    """Independent of EMBED_BATCH_ROWS, which is the database write batch."""

    def test_cpu_keeps_the_size_the_corpus_was_embedded_with(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=cpu_only_wheel())

        assert device.encode_batch_size(choice, 0) == 16

    def test_cuda_gets_a_larger_default(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=usable_gpu())

        assert device.encode_batch_size(choice, 0) == 32

    def test_an_explicit_size_wins_on_either_device(self) -> None:
        cpu = device.resolve(device.AUTO, torch_module=cpu_only_wheel())
        cuda = device.resolve(device.AUTO, torch_module=usable_gpu())

        assert device.encode_batch_size(cpu, 8) == 8
        assert device.encode_batch_size(cuda, 8) == 8

    def test_a_negative_size_is_refused(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=cpu_only_wheel())

        with pytest.raises(DeviceError):
            device.encode_batch_size(choice, -1)


class TestWhatTheRecordSays:
    def test_a_cpu_run_records_no_gpu_even_when_one_is_present(self) -> None:
        """The artifact says what ran, not what was available.

        This is the assertion that matters. A machine with a usable GPU that was
        asked for cpu must not leave a GPU name or a CUDA version in the record,
        or every future reader of that artifact will believe the vectors came off
        the card.
        """
        choice = device.resolve(device.CPU, torch_module=usable_gpu())

        recorded = choice.as_dict()

        assert recorded["device"] == device.CPU
        assert recorded["gpu_name"] is None
        assert recorded["cuda_version"] is None

    def test_a_cuda_run_records_the_versions_torch_reported(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=usable_gpu())

        recorded = choice.as_dict()

        assert recorded["device"] == device.CUDA
        assert recorded["torch_version"] == "2.13.0+cu124"
        assert recorded["cuda_version"] == "12.4"

    def test_the_record_carries_every_field_the_artifact_declares(self) -> None:
        choice = device.resolve(device.AUTO, torch_module=cpu_only_wheel())

        assert set(choice.as_dict()) == {
            "device",
            "torch_version",
            "cuda_version",
            "gpu_name",
            "precision",
        }
