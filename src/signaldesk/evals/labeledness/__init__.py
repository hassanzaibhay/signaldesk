"""Labeledness gold set: the sampling frame, the draw, and the annotation harness.

The gold set is curated by hand. Nothing in this package imports a model client,
and nothing in the annotation path imports Django or DuckDB either: the draw
reads the corpus once and writes every byte the annotator will see into a
committed manifest, so annotating needs one JSON file and nothing else.

``docs/annotation-guideline-labeledness.md`` is the normative document. This
package implements the mechanics it describes and does not define any of the
label semantics itself.
"""

from __future__ import annotations
