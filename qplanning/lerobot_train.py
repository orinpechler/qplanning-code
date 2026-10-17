"""LeRobot's training script, with :mod:`qplanning.lerobot_patches` applied first.

    python -m qplanning.lerobot_train <lerobot_train arguments>

Same arguments and behaviour as ``python -m lerobot.scripts.lerobot_train``; ``train-q``
launches this so that every rank of a (possibly distributed) run gets the patches.
"""

from __future__ import annotations

from qplanning.lerobot_patches import apply


def main() -> None:
    apply()
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()


if __name__ == "__main__":  # pragma: no cover
    main()
