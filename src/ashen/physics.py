"""Physical constants used by the JOREK profile-translation path.

Ports only the constants `boundary.py` and `profiles.py` actually consume from
`castor3d/util/physics.py:8-10`. That file also defines `kB`, used only by the
CASTOR3D-side `K_to_eV` -- not part of the JOREK path, so it is not ported here.
"""

from __future__ import annotations

import math

#: Elementary charge [C].
E_CHARGE = 1.6e-19

#: Vacuum permeability [H/m].
MU_0 = 4 * math.pi * 1e-7
