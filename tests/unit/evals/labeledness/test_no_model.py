"""No model, no database, and no network anywhere in the annotation path.

The import graph is walked statically over the source, not observed at runtime.
That matters: a runtime check only sees what a particular execution imported, so
an import buried inside a rarely-taken function passes it. Parsing every module
reachable from the entry points catches a function-level import as readily as a
top-level one.

WHAT THIS DOES NOT CLOSE, and it is deliberate that it is written down rather
than implied:

* a subprocess. ``subprocess.run(["python", "-c", "import anthropic; ..."])``
  is invisible to an import graph. Nothing here forbids it.
* a dynamic import on a computed name. ``importlib.import_module(name)`` where
  ``name`` is assembled at runtime cannot be resolved by parsing.

Both remain open. What is closed is every static import, at any nesting depth,
transitively through the first-party modules the annotation path reaches.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SOURCE_ROOT = Path("src")
PACKAGE_ROOT = SOURCE_ROOT / "signaldesk"

#: The entry points the annotator actually runs through.
ENTRY_MODULES = (
    "signaldesk.evals.labeledness.session",
    "signaldesk.evals.labeledness.render",
    "signaldesk.evals.labeledness.store",
    "signaldesk.evals.labeledness.manifest",
    "signaldesk.evals.labeledness.checkpoint",
    "signaldesk.evals.labeledness.schedule",
)

#: Top-level packages that must not appear anywhere in that graph.
FORBIDDEN_ROOTS = frozenset(
    {
        "anthropic",
        "openai",
        "cohere",
        "google",
        "mistralai",
        "ollama",
        "transformers",
        "torch",
        "sentence_transformers",
        "langchain",
        "litellm",
        "httpx",
        "requests",
        "aiohttp",
        "django",
        "duckdb",
        "polars",
        "psycopg",
        "celery",
        "redis",
        "bm25s",
    }
)

#: First-party modules that pull in a model client or the ORM. Named separately
#: because a first-party import looks innocuous and is exactly how one of these
#: would arrive.
FORBIDDEN_FIRST_PARTY = (
    "signaldesk.rag",
    "signaldesk.web",
    "signaldesk.analytics",
    "signaldesk.ingest",
    "signaldesk.evals.labeledness.draw",
)


def _module_path(module: str) -> Path | None:
    relative = Path(*module.split("."))
    candidate = SOURCE_ROOT / relative.with_suffix(".py")
    if candidate.is_file():
        return candidate
    package = SOURCE_ROOT / relative / "__init__.py"
    return package if package.is_file() else None


def _imports_of(path: Path) -> set[str]:
    """Every module named by an import statement, at any nesting depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _reachable() -> dict[str, set[str]]:
    """Transitive first-party closure of the entry points, with what each imports."""
    seen: dict[str, set[str]] = {}
    pending = list(ENTRY_MODULES)
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        path = _module_path(module)
        if path is None:
            continue
        imports = _imports_of(path)
        seen[module] = imports
        for name in imports:
            if name.startswith("signaldesk.") and name not in seen:
                pending.append(name)
    return seen


class TestTheAnnotationPathReachesNoModel:
    def test_the_package_root_is_where_this_test_thinks_it_is(self) -> None:
        """Guards the guard.

        Every assertion below is vacuous if the source cannot be found: an empty
        graph satisfies every "nothing in the graph" claim. This is the
        regression that would turn the whole file green while the property broke,
        so it is checked first.
        """
        assert PACKAGE_ROOT.is_dir(), f"source not found at {PACKAGE_ROOT.resolve()}"
        graph = _reachable()
        assert set(ENTRY_MODULES) <= set(graph)
        assert len(graph) > len(ENTRY_MODULES)

    def test_no_model_client_orm_or_http_library_is_reachable(self) -> None:
        graph = _reachable()
        offences: list[str] = []
        for module, imports in sorted(graph.items()):
            for name in sorted(imports):
                if name.split(".")[0] in FORBIDDEN_ROOTS:
                    offences.append(f"{module} imports {name}")
        assert offences == []

    def test_no_first_party_model_or_database_module_is_reachable(self) -> None:
        graph = _reachable()
        offences = [
            f"{module} imports {name}"
            for module, imports in sorted(graph.items())
            for name in sorted(imports)
            if any(name.startswith(prefix) for prefix in FORBIDDEN_FIRST_PARTY)
        ]
        assert offences == []

    def test_the_draw_module_is_not_reachable_from_the_harness(self) -> None:
        """The harness runs with no database; the draw needs one.

        Keeping them apart is what makes "no database at annotation time" a
        property of the code rather than a promise about how it is invoked.
        """
        assert "signaldesk.evals.labeledness.draw" not in _reachable()

    def test_the_detector_fires_on_a_planted_import(self, tmp_path: Path) -> None:
        """The other half of guarding the guard.

        A parser that silently returned nothing -- a changed AST node type, a
        swallowed SyntaxError -- would make every assertion above pass. Planting
        an import that must be found proves the walker still finds one.
        """
        planted = tmp_path / "planted.py"
        planted.write_text(
            "def later() -> None:\n    import anthropic\n    from django.db import models\n",
            encoding="utf-8",
        )
        found = _imports_of(planted)
        assert "anthropic" in found
        assert any(name.startswith("django") for name in found)
