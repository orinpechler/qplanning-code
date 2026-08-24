"""RoboTwin task inventory.

RoboTwin ships 50 bimanual tasks. Three of them cannot be evaluated with a *learned* policy:
their ``check_success()`` reads an attribute that is only ever assigned by the scripted
demonstration routine, so any policy that does not run that routine raises ``AttributeError``
at the end of the episode. They are excluded by default, leaving 47 evaluable
tasks -- not a filtering choice, an upstream limitation.
"""

from __future__ import annotations

ALL_TASKS: tuple[str, ...] = (
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle", "shake_bottle_horizontally", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
)

#: Tasks whose success check only works under the scripted demo policy.
BROKEN_TASKS: dict[str, str] = {
    "open_laptop": "check_success() reads self.arm_tag, which only play_once() assigns",
    "place_object_scale": "check_success() reads self.arm_tag, which only play_once() assigns",
    "put_object_cabinet": "check_success() reads self.arm_tag, which only play_once() assigns",
}

#: The 47 evaluable tasks.
PAPER_TASKS: tuple[str, ...] = tuple(t for t in ALL_TASKS if t not in BROKEN_TASKS)


def resolve_tasks(spec: str | list[str]) -> list[str]:
    """Turn a ``benchmark.tasks`` value into an explicit task list."""
    if isinstance(spec, list):
        unknown = [t for t in spec if t not in ALL_TASKS]
        if unknown:
            raise ValueError(f"unknown RoboTwin task(s): {unknown}")
        return list(spec)
    if spec == "paper47":
        return list(PAPER_TASKS)
    if spec == "all":
        return list(ALL_TASKS)
    if spec in ALL_TASKS:
        return [spec]
    raise ValueError(
        f"benchmark.tasks must be 'all', 'paper47', a task name, or a list; got {spec!r}"
    )
