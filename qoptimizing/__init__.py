"""Q-gradient action optimization: refine the policy's own chunk by ascending a frozen Q.

Not Q-Planning. There is no candidate weighting here: FastWAM draws its chunk exactly as in
the unguided baseline, and that chunk is then moved a fixed number of gradient-ascent steps
uphill on the Q-function before it is executed. Both networks stay frozen; the only tensor
that receives a gradient is the action chunk itself.

Importing :mod:`qplanning` first is deliberate -- it prepares the process environment before
anything can pull in LeRobot (see :mod:`qplanning._env_setup`).
"""

from __future__ import annotations

import qplanning  # noqa: F401
