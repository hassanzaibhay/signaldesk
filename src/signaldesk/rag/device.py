"""Which device a model runs on, decided once and recorded on what it produces.

The cuda half of this module has never executed. No GPU is reachable from this
project's container - the image installs a CPU-only torch wheel and compose maps
no device in - and continuous integration installs no torch at all. Device
selection, the refusal paths, the batch-size defaults and the recorded fields are
exercised; the cuda forward pass, the fp16 autocast and an OOM against a real
allocation failure are implemented and unit-tested against fakes and have never
run on hardware.

Nothing here imports torch at module scope. Every function takes the torch module
as an argument and one pragma'd line supplies the real one, so this file loads and
its logic runs in a job that has no torch - which is every continuous integration
job, including the one that enforces the coverage floor.

Three states have to stay apart, and torch alone cannot separate two of them:

``torch_built_without_cuda``
    ``torch.version.cuda`` is None. The wheel has no CUDA in it. Whether a card
    exists is not something this torch can answer, so it is not asserted either
    way from torch.
``cuda_build_present_but_unavailable``
    A CUDA build that cannot reach a device: no driver, or no passthrough into
    the container.
``cuda_available``
    A device torch can use.

``gpu_device_nodes_present`` is measured separately, off the filesystem, and is
what distinguishes "no card here" from "a card this process cannot use". It is
carried on every log line and named in every refusal, because a CPU fallback on a
machine that has a GPU is the case worth being loud about.

A fourth reason, ``cpu_requested``, describes a request rather than a detection:
it is what an explicit ``--device cpu`` resolves to, whatever the hardware is.

Two things are errors rather than fallbacks. Asking for cuda when cuda is
unavailable fails, because a request that cannot be honoured is not a request for
whatever is left. Running out of memory fails with the batch size that caused it,
with no smaller retry and no switch to the CPU: a run that quietly changed device
partway through would write an artifact describing two machines.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signaldesk.core.errors import SignalDeskError
from signaldesk.core.logging import get_logger

log = get_logger(__name__)

AUTO = "auto"
CPU = "cpu"
CUDA = "cuda"

#: What ``--device`` accepts. "auto" is a resolution, not a device.
REQUESTABLE = (AUTO, CPU, CUDA)

FP32 = "fp32"
FP16 = "fp16"

REASON_CUDA_AVAILABLE = "cuda_available"
REASON_NO_CUDA_BUILD = "torch_built_without_cuda"
REASON_UNAVAILABLE = "cuda_build_present_but_unavailable"
REASON_CPU_REQUESTED = "cpu_requested"

#: Texts per forward pass, by device, when the caller names no size.
#:
#: cpu is 16, which is what the committed corpus was embedded with and is not
#: changed here. cuda is 32: the only card this project has seen is a 4 GB
#: RTX 3050 Ti with a display attached, MedCPT-Article-Encoder is a
#: PubMedBERT-base at roughly 440 MB of fp32 weights before a single activation,
#: and 32 leaves real headroom on that budget while still doubling the CPU batch.
#: A constant with a stated reason rather than a number derived from free VRAM:
#: a size that changes with whatever else is on the card makes two runs of the
#: same command two different runs.
DEFAULT_ENCODE_BATCH = {CPU: 16, CUDA: 32}

#: Where a mapped-in GPU shows up inside a container. ``nvidia*`` is the usual
#: Linux case; ``dxg`` is the WSL2 compute node.
_DEV_ROOT = Path("/dev")


class DeviceError(SignalDeskError):
    """A device was asked for that cannot be given, or one ran out of memory."""


@dataclass(frozen=True, slots=True)
class TorchCapability:
    """What torch says about itself. Read once, in one place."""

    torch_version: str
    cuda_version: str | None
    cuda_available: bool
    gpu_count: int
    gpu_name: str | None


@dataclass(frozen=True, slots=True)
class DeviceChoice:
    """The resolved device, and everything a record needs to name it.

    ``cuda_version`` and ``gpu_name`` describe the device that was *used*, so
    they are None on a cpu run even when torch reported a usable card. The log
    line carries what torch saw; this carries what ran. An artifact that mixed
    the two would let a reader conclude the vectors came off a card that was
    merely present.
    """

    name: str
    reason: str
    torch_version: str
    cuda_version: str | None
    gpu_name: str | None
    precision: str

    def as_dict(self) -> dict[str, object]:
        """The fields an index artifact records under ``dense``."""
        return {
            "device": self.name,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "precision": self.precision,
        }


def _torch() -> Any:  # pragma: no cover - torch is absent from CI
    return importlib.import_module("torch")


def gpu_device_nodes_present(dev_root: Path | None = None) -> bool:
    """Whether a GPU is mapped into this container, read off the filesystem.

    Deliberately not ``nvidia-smi``: a subprocess whose absence means nothing in
    particular, in a container that may or may not carry the binary, answers a
    different question than "is there a device node here". This is a fact about
    the mount namespace and it is read as one.
    """
    root = dev_root if dev_root is not None else _DEV_ROOT
    if not root.is_dir():
        return False
    return any(root.glob("nvidia*")) or (root / "dxg").exists()


def capability(torch_module: Any) -> TorchCapability:
    """Read torch's own account of what it can do."""
    cuda_version = getattr(torch_module.version, "cuda", None)
    available = bool(torch_module.cuda.is_available())
    count = int(torch_module.cuda.device_count()) if available else 0
    name: str | None = None
    if available and count:
        name = str(torch_module.cuda.get_device_properties(0).name)
    return TorchCapability(
        torch_version=str(torch_module.__version__),
        cuda_version=str(cuda_version) if cuda_version is not None else None,
        cuda_available=available,
        gpu_count=count,
        gpu_name=name,
    )


def _detect(found: TorchCapability) -> tuple[str, str]:
    """The device torch could use on its own, and why."""
    if found.cuda_available:
        return CUDA, REASON_CUDA_AVAILABLE
    if found.cuda_version is None:
        return CPU, REASON_NO_CUDA_BUILD
    return CPU, REASON_UNAVAILABLE


def _refusal(reason: str, found: TorchCapability, *, nodes: bool) -> str:
    seen = (
        "a gpu device node is present, so a card is mapped in"
        if nodes
        else "no gpu device node is present, so no card is mapped into this container"
    )
    if reason == REASON_NO_CUDA_BUILD:
        detail = (
            f"torch {found.torch_version} was built without CUDA "
            f"(torch.version.cuda is None), so it cannot use a device even if one "
            f"is there. Separately: {seen}."
        )
    else:
        detail = (
            f"torch {found.torch_version} carries CUDA {found.cuda_version} but "
            f"torch.cuda.is_available() is false, so no device is reachable. "
            f"Separately: {seen}."
        )
    return (
        f"--device cuda was requested and cuda is unavailable ({reason}). {detail} "
        "This is refused rather than run on the cpu: an explicit device that is "
        "silently substituted produces an artifact naming hardware that did not "
        "run. Use --device auto to take whatever is available."
    )


def resolve(
    requested: str,
    *,
    fp16: bool = False,
    torch_module: Any | None = None,
    dev_root: Path | None = None,
) -> DeviceChoice:
    """Decide the device, log the decision and its reason, return the record.

    Every path through this function logs, cpu included, so a fallback on a
    machine that has a GPU cannot happen quietly.
    """
    if requested not in REQUESTABLE:
        message = f"unknown device {requested!r}; expected one of {', '.join(REQUESTABLE)}"
        raise DeviceError(message)

    found = capability(torch_module if torch_module is not None else _torch())
    nodes = gpu_device_nodes_present(dev_root)
    detected, detected_reason = _detect(found)

    if requested == CUDA and detected != CUDA:
        raise DeviceError(_refusal(detected_reason, found, nodes=nodes))

    if requested == CPU:
        name, reason = CPU, REASON_CPU_REQUESTED
    else:
        name, reason = detected, detected_reason

    if fp16 and name != CUDA:
        message = (
            "--fp16 was requested and the resolved device is cpu. Autocast on the "
            "cpu is a different dtype with different numerics, so this is refused "
            "rather than quietly ignored or quietly substituted. Drop --fp16, or "
            "resolve a cuda device."
        )
        raise DeviceError(message)

    choice = DeviceChoice(
        name=name,
        reason=reason,
        torch_version=found.torch_version,
        cuda_version=found.cuda_version if name == CUDA else None,
        gpu_name=found.gpu_name if name == CUDA else None,
        precision=FP16 if fp16 else FP32,
    )
    # Bound from `found` rather than from `choice`: the log says what was seen,
    # including a card that was available and not used, which is the thing a
    # reader of a cpu run's logs most needs to know.
    log.info(
        "rag.device.resolved",
        requested=requested,
        resolved=choice.name,
        reason=choice.reason,
        torch_version=found.torch_version,
        cuda_version=found.cuda_version,
        gpu_name=found.gpu_name,
        gpu_count=found.gpu_count,
        gpu_device_nodes_present=nodes,
        precision=choice.precision,
    )
    return choice


def autocast_context(torch_module: Any, choice: DeviceChoice) -> AbstractContextManager[None]:
    """The context one forward pass runs in.

    ``nullcontext`` on the fp32 path rather than an autocast constructed with
    ``enabled=False``. The two are meant to be equivalent, but only one of them
    is provably nothing, and the fp32 path is the one that produced every vector
    the index already holds.
    """
    if choice.precision != FP16:
        return nullcontext()
    context: AbstractContextManager[None] = torch_module.autocast(
        device_type=choice.name, dtype=torch_module.float16
    )
    return context


def _is_torch_oom(error: BaseException) -> bool:
    """torch's own OutOfMemoryError, and not another library's by that name.

    Matched by name and module rather than by class, because this module has no
    torch to import and CI has none to compare against.
    """
    kind = type(error)
    return kind.__name__ == "OutOfMemoryError" and kind.__module__.split(".")[0] == "torch"


@contextmanager
def surfacing_oom(*, batch_size: int, device: str) -> Iterator[None]:
    """Turn a device OOM into an error that names the batch size that caused it.

    No smaller retry and no fall back to the cpu. Both would let a run finish
    while describing itself as something it was not: half its work at one batch
    size and half at another, or half on one device and half on another. The
    caller lowers --batch-size and runs it again, and then the record says one
    thing.
    """
    try:
        yield
    except Exception as error:
        if not _is_torch_oom(error):
            raise
        message = (
            f"the {device} device ran out of memory at an encode batch size of "
            f"{batch_size}. Re-run with a smaller --batch-size. This is not "
            "retried automatically at a lower size and does not fall back to the "
            "cpu, because a run that changed either partway through would record "
            "one set of parameters for two different pieces of work."
        )
        raise DeviceError(message) from error


def encode_batch_size(choice: DeviceChoice, requested: int) -> int:
    """Texts per forward pass: what was asked for, or the device's default.

    Zero means "the default for this device", matching --threads. This is the
    encode batch and nothing else; the database write batch is EMBED_BATCH_ROWS
    in rag.index.corpus and is not derived from it.
    """
    if requested < 0:
        message = f"batch size must not be negative, got {requested}"
        raise DeviceError(message)
    if requested:
        return requested
    return DEFAULT_ENCODE_BATCH[choice.name]
