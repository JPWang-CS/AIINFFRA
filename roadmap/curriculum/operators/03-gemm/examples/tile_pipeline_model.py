"""纯 CPU 教学模型：GEMM 输出归属、有限 slot 生命周期和资源账本。

这不是 GPU 模拟器，也不表示真实的并发、调度或吞吐；它只把几个常见
的教学约束写成可检查的离散模型。
"""

from __future__ import annotations

import unittest


def warp_output_ownership(block_m: int, block_n: int, warp_m: int, warp_n: int):
    """返回每个 warp 覆盖的输出坐标，并检查 exact cover。"""
    values = (block_m, block_n, warp_m, warp_n)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("tile dimensions must be positive integers")
    if block_m % warp_m or block_n % warp_n:
        raise ValueError("block tile must be divisible by warp tile")

    warp_rows = block_m // warp_m
    warp_cols = block_n // warp_n
    ownership = {}
    for warp_id in range(warp_rows * warp_cols):
        warp_row, warp_col = divmod(warp_id, warp_cols)
        rows = range(warp_row * warp_m, (warp_row + 1) * warp_m)
        cols = range(warp_col * warp_n, (warp_col + 1) * warp_n)
        ownership[warp_id] = [(m, n) for m in rows for n in cols]

    all_coordinates = [coordinate for coordinates in ownership.values() for coordinate in coordinates]
    if len(all_coordinates) != block_m * block_n or len(set(all_coordinates)) != block_m * block_n:
        raise AssertionError("warp ownership is not an exact cover")
    return {
        "warps": ownership,
        "warp_count": warp_rows * warp_cols,
        "accumulator_per_lane_avg": warp_m * warp_n / 32,
    }


def validate_trace(events, stages=None):
    """检查发行顺序、FIFO 读取/回收和 slot 生命周期；copydone 可乱序。"""
    if stages is not None and (not isinstance(stages, int) or stages <= 0):
        raise ValueError("stages must be a positive integer when supplied")
    states = {}
    issued = set()
    action_counts = {}
    next_issue = 0
    next_consume = 0
    slot_generations = {}
    for event in events:
        if not isinstance(event, tuple) or len(event) != 4:
            raise AssertionError("event must be (tile, slot, generation, action)")
        tile, slot, generation, action = event
        if not all(isinstance(value, int) for value in (tile, slot, generation)):
            raise AssertionError("tile, slot and generation must be integers")
        if tile < 0 or slot < 0 or generation < 0:
            raise AssertionError("tile, slot and generation must be non-negative")
        if action not in {"prefill", "copydone", "read", "release"}:
            raise AssertionError(f"unknown action: {action}")
        state = states.get(slot, ("EMPTY", None, None))
        current_state, current_tile, current_generation = state
        if action == "prefill":
            if tile != next_issue or tile in issued:
                raise AssertionError("prefill must issue each tile exactly once in order")
            if current_state != "EMPTY":
                raise AssertionError("prefill overwrites a slot that was not released")
            if stages is not None:
                expected_slot = tile % stages
                expected_generation = tile // stages
                if (slot, generation) != (expected_slot, expected_generation):
                    raise AssertionError("slot/generation does not match tile and stages")
            elif generation != slot_generations.get(slot, 0):
                raise AssertionError("slot generation must increase on each reuse")
            slot_generations[slot] = generation + 1
            issued.add(tile)
            next_issue += 1
            states[slot] = ("IN_FLIGHT", tile, generation)
        elif action == "copydone":
            if current_state != "IN_FLIGHT" or (current_tile, current_generation) != (tile, generation):
                raise AssertionError("copydone must follow the matching prefill")
            states[slot] = ("READY", tile, generation)
        elif action == "read":
            if tile != next_consume:
                raise AssertionError("read must consume tiles in FIFO order")
            if current_state != "READY" or (current_tile, current_generation) != (tile, generation):
                raise AssertionError("read requires a completed copy")
            states[slot] = ("READING", tile, generation)
        else:  # release
            if tile != next_consume:
                raise AssertionError("release must consume tiles in FIFO order")
            if current_state != "READING" or (current_tile, current_generation) != (tile, generation):
                raise AssertionError("release must follow read for the same tile")
            states[slot] = ("EMPTY", None, None)
            next_consume += 1
        action_counts[tile] = action_counts.get(tile, 0) + 1
    if any(state[0] != "EMPTY" for state in states.values()):
        raise AssertionError("trace ended with unreleased slots")
    if next_issue != next_consume or any(count != 4 for count in action_counts.values()):
        raise AssertionError("every issued tile must have exactly four ordered events")
    return True


def pipeline_protocol(tiles: int, stages: int):
    """离散地模拟环形 slot：EMPTY -> IN_FLIGHT -> READY -> READING -> EMPTY。"""
    if not isinstance(tiles, int) or tiles < 0:
        raise ValueError("tiles must be a non-negative integer")
    if not isinstance(stages, int) or stages <= 0:
        raise ValueError("stages must be a positive integer")

    events = []
    slots = [None] * stages

    def consume(slot):
        tile, generation = slots[slot]
        events.extend(
            (
                (tile, slot, generation, "copydone"),
                (tile, slot, generation, "read"),
                (tile, slot, generation, "release"),
            )
        )
        slots[slot] = None

    initial = min(stages, tiles)
    for tile in range(initial):
        slot = tile % stages
        generation = tile // stages
        events.append((tile, slot, generation, "prefill"))
        slots[slot] = (tile, generation)

    for tile in range(tiles):
        slot = tile % stages
        consume(slot)
        next_tile = tile + stages
        if next_tile < tiles:
            generation = next_tile // stages
            events.append((next_tile, slot, generation, "prefill"))
            slots[slot] = (next_tile, generation)

    validate_trace(events, stages)
    return events


def resource_ledger(
    block_m: int,
    block_n: int,
    reduction_tile: int,
    stages: int,
    input_bytes: int,
    warps: int,
):
    """返回教学资源估算；acc_fp32_avg 不是 compiler 的寄存器计数。"""
    values = (block_m, block_n, reduction_tile, stages, input_bytes, warps)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("resource parameters must be positive integers")
    shared = stages * reduction_tile * (block_m + block_n) * input_bytes
    acc_fp32_avg = block_m * block_n / (warps * 32)
    return {
        "shared_bytes": shared,
        "acc_fp32_avg": acc_fp32_avg,
        "acc_fp32_avg_note": "mean accumulator count per lane; not compiler register count",
    }


class ModelTests(unittest.TestCase):
    def test_ownership_exact_cover(self):
        result = warp_output_ownership(128, 128, 64, 32)
        self.assertEqual(result["warp_count"], 8)
        self.assertEqual(result["accumulator_per_lane_avg"], 64)
        self.assertEqual(sum(map(len, result["warps"].values())), 128 * 128)

    def test_pipeline_sizes_and_drain(self):
        for tiles in (0, 1, 2, 7, 10):
            for stages in (1, 2, 3):
                events = pipeline_protocol(tiles, stages)
                self.assertTrue(validate_trace(events, stages))
                self.assertEqual({event[0] for event in events}, set(range(tiles)))
                self.assertEqual(
                    [event[0] for event in events if event[3] == "read"],
                    list(range(tiles)),
                )
                self.assertTrue(all(sum(event[0] == tile for event in events) == 4 for tile in range(tiles)))

    def test_pipeline_rejects_bad_order(self):
        with self.assertRaises(AssertionError):
            validate_trace([(0, 0, 0, "read")])
        with self.assertRaises(AssertionError):
            validate_trace([(0, 0, 0, "prefill"), (1, 0, 0, "prefill")])
        with self.assertRaises(AssertionError):
            validate_trace([(0, 0, 0, "prefill"), (0, 0, 0, "copydone"), (0, 0, 0, "release")])
        with self.assertRaises(AssertionError):
            validate_trace(
                [(0, 0, 0, "prefill"), (1, 1, 0, "prefill"),
                 (1, 1, 0, "copydone"), (1, 1, 0, "read"), (1, 1, 0, "release"),
                 (0, 0, 0, "copydone"), (0, 0, 0, "read"), (0, 0, 0, "release")],
                stages=2,
            )
        with self.assertRaises(AssertionError):
            validate_trace(
                [(0, 0, 0, "prefill"), (0, 0, 0, "prefill")], stages=1
            )
        with self.assertRaises(AssertionError):
            validate_trace(
                [(0, 0, 1, "prefill"), (0, 0, 1, "copydone"),
                 (0, 0, 1, "read"), (0, 0, 1, "release")], stages=1
            )

    def test_completion_can_be_out_of_order_but_read_is_fifo(self):
        events = [
            (0, 0, 0, "prefill"),
            (1, 1, 0, "prefill"),
            (1, 1, 0, "copydone"),
            (0, 0, 0, "copydone"),
            (0, 0, 0, "read"),
            (0, 0, 0, "release"),
            (1, 1, 0, "read"),
            (1, 1, 0, "release"),
        ]
        self.assertTrue(validate_trace(events, stages=2))

    def test_ledger(self):
        self.assertEqual(resource_ledger(128, 128, 32, 3, 2, 8)["shared_bytes"], 49152)
        self.assertEqual(resource_ledger(128, 128, 32, 3, 2, 8)["acc_fp32_avg"], 64)
        self.assertEqual(resource_ledger(128, 128, 32, 4, 2, 8)["shared_bytes"], 65536)


if __name__ == "__main__":
    unittest.main()
