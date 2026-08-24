"""LIBERO suite inventory.

Each suite carries its own episode budget, applied by the environment when a config leaves
``benchmark.episode_length`` as ``null``.
"""

from __future__ import annotations

SUITES: tuple[str, ...] = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

#: Human-readable names for LIBERO-10, used in per-episode result files.
LIBERO_10_TASK_NAMES: dict[int, str] = {
    0: "put_alphabet_soup_and_tomato_sauce_in_basket",
    1: "put_cream_cheese_and_butter_in_basket",
    2: "turn_on_stove_and_put_moka_pot",
    3: "put_black_bowl_in_bottom_drawer_and_close",
    4: "put_white_mug_left_plate_yellow_mug_right_plate",
    5: "pick_up_book_and_place_in_caddy",
    6: "put_white_mug_on_plate_and_chocolate_pudding_right",
    7: "put_alphabet_soup_and_cream_cheese_in_basket",
    8: "put_both_moka_pots_on_stove",
    9: "put_yellow_white_mug_in_microwave_and_close",
}


def task_name(suite: str, task_id: int, fallback: str | None = None) -> str:
    """A stable name for a task, for result files."""
    if suite == "libero_10" and task_id in LIBERO_10_TASK_NAMES:
        return LIBERO_10_TASK_NAMES[task_id]
    return fallback or f"{suite}_task_{task_id}"
