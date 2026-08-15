"""The Haldane-Anscombe continuity correction, and where it is allowed to apply.

Adding 0.5 to every cell of a 2x2 table makes the odds ratio and the relative
risk finite when a cell is empty. It also pulls every estimate toward the null,
by an amount that grows as the counts shrink. Applied unconditionally it would
bias the entire signal table; applied only where a cell is zero it rescues the
rows that would otherwise be infinite or undefined and leaves the rest alone.

So this module exposes one function, it corrects **per row**, and it returns the
mask of rows it touched so the flag can travel with the result to the UI and the
artifact.

**Where it is applied, exhaustively:**

* ROR point estimate and interval - yes, on rows with a zero cell.
* PRR point estimate and interval - yes, on rows with a zero cell.
* PRR chi-squared - **no**. Yates' continuity correction is already a
  small-cell adjustment on that statistic, and stacking the two shrinks the
  same deviation twice. The chi-squared is computed from the observed counts.
* BCPNN - **no**. Its prior already shrinks small counts, which is the reason
  to run it, and a pseudo-count on top would shrink twice.
* MGPS - **no**, for the same reason.
"""

from __future__ import annotations

import numpy as np

from signaldesk.stats.types import BoolArray, Contingency, FloatArray

#: Anscombe (1956) and Haldane (1956). Half a case in each cell.
CORRECTION = 0.5


def haldane_anscombe(table: Contingency) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return the cells as floats, with 0.5 added to rows holding a zero cell.

    Rows with no zero cell come back numerically unchanged, so a caller can use
    the output unconditionally without a second code path.
    """
    adjustment = np.where(table.has_zero_cell, CORRECTION, 0.0)
    return (
        table.a.astype(np.float64) + adjustment,
        table.b.astype(np.float64) + adjustment,
        table.c.astype(np.float64) + adjustment,
        table.d.astype(np.float64) + adjustment,
    )


def corrected_rows(table: Contingency) -> BoolArray:
    """Which rows ``haldane_anscombe`` adjusted. Stored on every result row."""
    return table.has_zero_cell
