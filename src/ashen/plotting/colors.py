"""Colouring field lines by their starting flux surface.

The legacy code indexed a hand-written 135-line list literal
(``contrast_colors``, ``data_jorek.py:27-135`` -- the same ~19-colour block
copy-pasted four times with drifting values) positionally by a field line's
index in the scan. The new cache has no such index: it is a flat dict keyed
by starting position (see :mod:`ashen.diagnostics.poincare_cache`), so
"index 7" is not a stable or even meaningful concept once a scan has been
extended incrementally.

Colour is assigned from **the line's actual ``psi_n``** instead, through a
perceptually uniform colormap (default), which is also what
``gather_profiles.py:142`` already does for time series and is the only
sensible colouring in the legacy tree. A colour then means the same radial
position across every plot and every case, a legend becomes an honest
colourbar, and there is no line count that can overflow it.

A small discrete palette is also provided for a categorical look, deduplicated
from the legacy list (its four copies did not actually agree with each other).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["DISCRETE_PALETTE", "PsiColorer", "colorer"]

#: Deduplicated from contrast_colors (data_jorek.py:27-135); cycles past 19.
DISCRETE_PALETTE: tuple[str, ...] = (
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#FFFF33", "#A65628", "#F781BF", "#999999", "#66C2A5",
    "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F",
    "#E5C494", "#B3B3B3", "#1B9E77", "#4B0082",
)


@dataclass(frozen=True)
class PsiColorer:
    """Maps a ``psi_n`` value to a colour, over a fixed ``[vmin, vmax]``.

    Built once per plot from the full set of ``psi_n`` values so that colour
    is consistent across every line drawn on the same axes -- as opposed to
    re-normalising per call, which would make the same physical surface a
    different colour depending on what else happened to be plotted alongside
    it.
    """

    vmin: float
    vmax: float
    cmap: str = "viridis"
    discrete: bool = False

    def __call__(self, psi_n: float):
        if self.discrete:
            # Stable across a run: bucketed by position in [vmin, vmax]
            # rather than by insertion order, so the same psi_n always gets
            # the same colour regardless of what else is in the plot.
            span = max(self.vmax - self.vmin, 1e-12)
            idx = int((psi_n - self.vmin) / span * len(DISCRETE_PALETTE))
            idx = min(max(idx, 0), len(DISCRETE_PALETTE) - 1)
            return DISCRETE_PALETTE[idx]

        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        return plt.get_cmap(self.cmap)(norm(psi_n))

    def scalar_mappable(self):
        """For attaching a colourbar -- meaningless in discrete mode."""
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        return ScalarMappable(
            norm=Normalize(vmin=self.vmin, vmax=self.vmax), cmap=plt.get_cmap(self.cmap)
        )


def colorer(psi_n_values: Sequence[float], *, cmap: str = "viridis", discrete: bool = False) -> PsiColorer:
    """Build a :class:`PsiColorer` spanning the given ``psi_n`` values."""
    values = list(psi_n_values)
    if not values:
        return PsiColorer(vmin=0.0, vmax=1.0, cmap=cmap, discrete=discrete)
    return PsiColorer(vmin=min(values), vmax=max(values), cmap=cmap, discrete=discrete)
