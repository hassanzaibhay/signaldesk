"""A stand-in for the torch namespace, holding only what the resolver reads.

Continuous integration installs the dev group and never the ``ml`` extra, so
torch does not exist in the job that runs these tests and there is nothing to
monkeypatch. ``rag.device`` takes the torch module as an argument for exactly
this reason, and these are what it gets handed. Same substitution
monkeypatching would make, and the only one that runs in CI.

The three factories at the bottom are the three states the resolver has to keep
apart, named for what they are rather than for their flags.
"""

from __future__ import annotations

from typing import Any


class Properties:
    def __init__(self, name: str) -> None:
        self.name = name


class Cuda:
    def __init__(self, *, available: bool, count: int, gpu_name: str) -> None:
        self._available = available
        self._count = count
        self._gpu_name = gpu_name

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_properties(self, index: int) -> Properties:
        return Properties(self._gpu_name)


class Version:
    def __init__(self, cuda: str | None) -> None:
        self.cuda = cuda


class FakeTorch:
    """Records every autocast it was asked to build, so a test can assert none."""

    def __init__(
        self,
        *,
        cuda_build: str | None,
        available: bool = False,
        count: int = 0,
        gpu_name: str = "",
        version: str = "2.13.0+cpu",
    ) -> None:
        self.__version__ = version
        self.version = Version(cuda_build)
        self.cuda = Cuda(available=available, count=count, gpu_name=gpu_name)
        self.float16 = "float16-sentinel"
        self.autocast_calls: list[dict[str, Any]] = []

    def autocast(self, **kwargs: Any) -> str:
        self.autocast_calls.append(kwargs)
        return "autocast-sentinel"


def cpu_only_wheel() -> FakeTorch:
    """What this project's container reports today: a torch built without CUDA."""
    return FakeTorch(cuda_build=None)


def unusable_gpu() -> FakeTorch:
    """A CUDA build that cannot reach a device: no driver, or no passthrough."""
    return FakeTorch(cuda_build="12.4", available=False, version="2.13.0+cu124")


def usable_gpu() -> FakeTorch:
    return FakeTorch(
        cuda_build="12.4",
        available=True,
        count=1,
        gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        version="2.13.0+cu124",
    )
