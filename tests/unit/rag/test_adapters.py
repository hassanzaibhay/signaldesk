"""What can be checked about the MedCPT adapters without torch.

Importing the module is safe: every transformers and torch import inside it is
inside a method, so the file loads anywhere. What cannot be checked here is any
behaviour, because behaviour needs the weights - and that is the point of the
size guard below.

The guard is the mechanical half of "adapters stay thin". The constraint is that
batching, normalisation, the dimension assertion, the pair construction and
resumption live in covered code; the way that constraint fails is by logic
drifting into these classes one convenience at a time, each defensible on its
own. A number that has to be edited to grow makes that visible in review.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from signaldesk.rag import adapters
from signaldesk.rag.embed import CrossEncoder, DocumentEncoder, Encoder

pytestmark = pytest.mark.unit

#: Executable statements allowed across every adapter class combined. Not a
#: style preference: this is the exact count of lines continuous integration
#: will never run, and it should only ever go down.
MAX_UNCOVERABLE_STATEMENTS = 30

SOURCE = Path(inspect.getfile(adapters))


def _statements(node: ast.ClassDef) -> int:
    """Executable statements in a class, ignoring docstrings and definitions."""
    count = 0
    for statement in ast.walk(node):
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.arguments):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, ast.stmt):
            count += 1
    return count


def _classes() -> list[ast.ClassDef]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return [node for node in tree.body if isinstance(node, ast.ClassDef)]


class TestTheUncoverableSurface:
    def test_it_stays_within_budget(self) -> None:
        total = sum(_statements(node) for node in _classes())

        assert total <= MAX_UNCOVERABLE_STATEMENTS, (
            f"the adapters hold {total} executable statements, over the "
            f"{MAX_UNCOVERABLE_STATEMENTS} budget. Continuous integration runs none "
            "of them, so something that decides anything has moved somewhere it "
            "cannot be tested. Move it to rag.embed or rag.index.corpus."
        )

    def test_every_class_is_excluded_from_coverage(self) -> None:
        """An unmarked class would silently drag the whole project's coverage down."""
        source = SOURCE.read_text(encoding="utf-8").splitlines()

        for node in _classes():
            assert "pragma: no cover" in source[node.lineno - 1], (
                f"{node.name} is not marked; CI cannot execute it and would count "
                "its lines as missed."
            )

    def test_torch_and_transformers_are_imported_lazily(self) -> None:
        """The module has to import where neither is installed, which is CI."""
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        top_level = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
        names = {
            alias.name.split(".")[0]
            for node in top_level
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in top_level
            if isinstance(node, ast.ImportFrom) and node.module
        }

        assert "torch" not in names
        assert "transformers" not in names


class TestTheProtocols:
    """The adapters have to satisfy the interfaces the covered code is written against."""

    @staticmethod
    def _members(protocol: type) -> set[str]:
        return {name for name in protocol.__protocol_attrs__ if not name.startswith("_")}

    @pytest.mark.parametrize(
        ("adapter", "protocol"),
        [
            (adapters.MedCptArticleEncoder, DocumentEncoder),
            (adapters.MedCptQueryEncoder, Encoder),
            (adapters.MedCptCrossEncoder, CrossEncoder),
        ],
    )
    def test_each_adapter_carries_every_member_its_protocol_declares(
        self, adapter: type, protocol: type
    ) -> None:
        """Structural, because issubclass does not work on a protocol with
        properties and the adapters cannot be instantiated without torch."""
        missing = {name for name in self._members(protocol) if not hasattr(adapter, name)}

        assert not missing, f"{adapter.__name__} is missing {sorted(missing)}"

    def test_the_two_encoder_roles_take_different_input(self) -> None:
        """One takes pairs and one takes texts; the parameter names say which."""
        article = inspect.signature(adapters.MedCptArticleEncoder.encode)
        query = inspect.signature(adapters.MedCptQueryEncoder.encode)

        assert "pairs" in article.parameters
        assert "texts" in query.parameters

    def test_the_query_and_article_encoders_are_separate_classes(self) -> None:
        """MedCPT ships two checkpoints trained together for different jobs.

        One class taking both roles is how the wrong model gets passed to the
        wrong side, which is the defect this split was made to prevent.
        """
        assert adapters.MedCptArticleEncoder is not adapters.MedCptQueryEncoder

    def test_the_positional_limit_matches_the_published_config(self) -> None:
        """All three checkpoints declare max_position_embeddings of 512."""
        assert adapters.MAX_LENGTH == 512


class TestTheDeviceIsChosenByTheCaller:
    """Every adapter runs where it is told, and is told explicitly.

    The alternative is a default of "cpu" inside the adapter, which would make a
    caller that forgot to resolve a device look identical in the record to one
    that resolved cpu deliberately. A required argument makes the omission a
    TypeError at the call site instead of a wrong line in an artifact.
    """

    @pytest.mark.parametrize(
        "adapter",
        [
            adapters.MedCptArticleEncoder,
            adapters.MedCptQueryEncoder,
            adapters.MedCptCrossEncoder,
        ],
    )
    def test_each_adapter_takes_a_device(self, adapter: type) -> None:
        parameter = inspect.signature(adapter.__init__).parameters.get("device")

        assert parameter is not None, f"{adapter.__name__} cannot be pointed at a device"

    @pytest.mark.parametrize(
        "adapter",
        [
            adapters.MedCptArticleEncoder,
            adapters.MedCptQueryEncoder,
            adapters.MedCptCrossEncoder,
        ],
    )
    def test_the_device_has_no_default(self, adapter: type) -> None:
        parameter = inspect.signature(adapter.__init__).parameters["device"]

        assert parameter.default is inspect.Parameter.empty, (
            f"{adapter.__name__} defaults its device, so a caller that never "
            "resolved one produces a record indistinguishable from a deliberate "
            "cpu run"
        )

    @pytest.mark.parametrize(
        "adapter",
        [
            adapters.MedCptArticleEncoder,
            adapters.MedCptQueryEncoder,
            adapters.MedCptCrossEncoder,
        ],
    )
    def test_the_device_is_keyword_only(self, adapter: type) -> None:
        parameter = inspect.signature(adapter.__init__).parameters["device"]

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
