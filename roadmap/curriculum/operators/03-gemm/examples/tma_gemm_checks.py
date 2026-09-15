"""CPU-only protocol model for a two-input TMA GEMM barrier.

This checks the software-visible protocol contract: 32 distinct thread
arrivals and completion of the expected 512-byte A and 512-byte B transfers.
Transfer completions may be observed before or after thread arrivals. This is
not a simulation of a hardware barrier counter, TMA engine, or GPU execution.
"""

import unittest


THREAD_COUNT = 32
TRANSFER_BYTES = 512
INPUTS = ("A", "B")


class ProtocolError(ValueError):
    """Raised when an event violates the modeled barrier protocol."""


class TmaGemmSlot:
    """One reusable protocol slot with strictly increasing generations."""

    def __init__(self):
        self._next_generation = 0
        self._generation = None
        self._arrivals = set()
        self._completions = set()
        self._consumed = False

    def reserve(self, generation):
        if type(generation) is not int or generation < 0:
            raise ProtocolError("generation must be a non-negative integer")
        if self._generation is not None:
            raise ProtocolError("slot reused before release")
        if generation != self._next_generation:
            raise ProtocolError(
                f"stale generation {generation}; expected {self._next_generation}"
            )
        self._generation = generation
        self._arrivals.clear()
        self._completions.clear()
        self._consumed = False

    def _check_generation(self, generation):
        if self._generation is None:
            raise ProtocolError("slot is inactive; reserve a generation first")
        if type(generation) is not int or generation < 0:
            raise ProtocolError("generation must be a non-negative integer")
        if generation != self._generation:
            raise ProtocolError(
                f"stale generation {generation}; active generation is "
                f"{self._generation}"
            )

    def arrive(self, thread_id, generation):
        self._check_generation(generation)
        if type(thread_id) is not int or not 0 <= thread_id < THREAD_COUNT:
            raise ProtocolError(f"thread id must be in 0..{THREAD_COUNT - 1}")
        if thread_id in self._arrivals:
            raise ProtocolError(f"duplicate arrival from thread {thread_id}")
        self._arrivals.add(thread_id)

    def complete(self, input_name, byte_count, generation):
        self._check_generation(generation)
        if input_name not in INPUTS:
            raise ProtocolError(f"unknown input {input_name!r}; expected A or B")
        if input_name in self._completions:
            raise ProtocolError(f"duplicate completion for input {input_name}")
        if byte_count != TRANSFER_BYTES:
            raise ProtocolError(
                f"input {input_name} transferred {byte_count} bytes; "
                f"expected {TRANSFER_BYTES}"
            )
        self._completions.add(input_name)

    def is_ready(self, generation):
        self._check_generation(generation)
        return (
            len(self._arrivals) == THREAD_COUNT
            and self._completions == set(INPUTS)
        )

    def consume(self, generation):
        if not self.is_ready(generation):
            raise ProtocolError("slot is not ready: need 32 arrivals and A/B completion")
        if self._consumed:
            raise ProtocolError("generation already consumed")
        self._consumed = True

    def release(self, generation):
        self._check_generation(generation)
        if not self._consumed:
            raise ProtocolError("cannot release before consuming a ready generation")
        self._generation = None
        self._next_generation += 1


def arrive_all(slot, generation):
    for thread_id in range(THREAD_COUNT):
        slot.arrive(thread_id, generation)


class TmaGemmProtocolTests(unittest.TestCase):
    def test_a_then_b_completions(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        arrive_all(slot, 0)
        slot.complete("A", TRANSFER_BYTES, 0)
        self.assertFalse(slot.is_ready(0))
        slot.complete("B", TRANSFER_BYTES, 0)
        self.assertTrue(slot.is_ready(0))
        slot.consume(0)
        slot.release(0)

    def test_b_then_a_completions(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        arrive_all(slot, 0)
        slot.complete("B", TRANSFER_BYTES, 0)
        self.assertFalse(slot.is_ready(0))
        slot.complete("A", TRANSFER_BYTES, 0)
        self.assertTrue(slot.is_ready(0))

    def test_threads_arrive_before_copies(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        arrive_all(slot, 0)
        self.assertFalse(slot.is_ready(0))
        slot.complete("A", TRANSFER_BYTES, 0)
        slot.complete("B", TRANSFER_BYTES, 0)
        self.assertTrue(slot.is_ready(0))

    def test_copies_complete_before_threads(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        slot.complete("A", TRANSFER_BYTES, 0)
        slot.complete("B", TRANSFER_BYTES, 0)
        self.assertFalse(slot.is_ready(0))
        arrive_all(slot, 0)
        self.assertTrue(slot.is_ready(0))

    def test_missing_b_keeps_slot_not_ready(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        arrive_all(slot, 0)
        slot.complete("A", TRANSFER_BYTES, 0)
        self.assertFalse(slot.is_ready(0))
        with self.assertRaisesRegex(ProtocolError, "not ready"):
            slot.consume(0)

    def test_duplicate_arrival_and_completion_are_rejected(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        slot.arrive(7, 0)
        with self.assertRaisesRegex(ProtocolError, "duplicate arrival"):
            slot.arrive(7, 0)
        slot.complete("A", TRANSFER_BYTES, 0)
        with self.assertRaisesRegex(ProtocolError, "duplicate completion"):
            slot.complete("A", TRANSFER_BYTES, 0)

    def test_wrong_bytes_and_stale_generation_are_rejected(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        with self.assertRaisesRegex(ProtocolError, "expected 512"):
            slot.complete("A", TRANSFER_BYTES - 1, 0)
        with self.assertRaisesRegex(ProtocolError, "stale generation"):
            slot.arrive(0, 1)

    def test_reuse_before_release_is_rejected(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        with self.assertRaisesRegex(ProtocolError, "before release"):
            slot.reserve(0)

    def test_barrier_operations_rejected_when_slot_is_inactive(self):
        slot = TmaGemmSlot()
        inactive_operations = (
            lambda: slot.arrive(0, 0),
            lambda: slot.arrive(0, None),
            lambda: slot.complete("A", TRANSFER_BYTES, 0),
            lambda: slot.complete("A", TRANSFER_BYTES, None),
            lambda: slot.consume(0),
            lambda: slot.consume(None),
            lambda: slot.release(0),
            lambda: slot.release(None),
        )
        for operation in inactive_operations:
            with self.subTest(state="before reserve", operation=operation):
                with self.assertRaisesRegex(ProtocolError, "inactive"):
                    operation()

        slot.reserve(0)
        arrive_all(slot, 0)
        slot.complete("A", TRANSFER_BYTES, 0)
        slot.complete("B", TRANSFER_BYTES, 0)
        slot.consume(0)
        slot.release(0)
        for operation in inactive_operations:
            with self.subTest(state="after release", operation=operation):
                with self.assertRaisesRegex(ProtocolError, "inactive"):
                    operation()

    def test_old_generation_rejected_after_slot_is_reserved_again(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        arrive_all(slot, 0)
        slot.complete("A", TRANSFER_BYTES, 0)
        slot.complete("B", TRANSFER_BYTES, 0)
        slot.consume(0)
        slot.release(0)
        slot.reserve(1)

        stale_operations = (
            lambda: slot.arrive(0, 0),
            lambda: slot.complete("A", TRANSFER_BYTES, 0),
            lambda: slot.consume(0),
            lambda: slot.release(0),
        )
        for operation in stale_operations:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ProtocolError, "stale generation"):
                    operation()

    def test_invalid_generation_values_are_rejected(self):
        for generation in (None, True, False, -1, 0.0):
            with self.subTest(generation=generation):
                with self.assertRaisesRegex(ProtocolError, "non-negative integer"):
                    TmaGemmSlot().reserve(generation)

    def test_invalid_active_generation_values_are_rejected(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        for generation in (None, False, 0.0):
            operations = (
                lambda: slot.arrive(0, generation),
                lambda: slot.complete("A", TRANSFER_BYTES, generation),
                lambda: slot.consume(generation),
                lambda: slot.release(generation),
            )
            for operation in operations:
                with self.subTest(generation=generation, operation=operation):
                    with self.assertRaisesRegex(ProtocolError, "non-negative integer"):
                        operation()

    def test_none_and_boolean_thread_ids_are_rejected(self):
        slot = TmaGemmSlot()
        slot.reserve(0)
        for thread_id in (None, True, False):
            with self.subTest(thread_id=thread_id):
                with self.assertRaisesRegex(ProtocolError, "thread id"):
                    slot.arrive(thread_id, 0)

    def test_two_slots_round_robin_for_five_generations(self):
        slots = [TmaGemmSlot(), TmaGemmSlot()]
        for generation in range(5):
            slot = slots[generation % 2]
            slot.reserve(generation // 2)
            slot.complete("A", TRANSFER_BYTES, generation // 2)
            slot.complete("B", TRANSFER_BYTES, generation // 2)
            arrive_all(slot, generation // 2)
            self.assertTrue(slot.is_ready(generation // 2))
            slot.consume(generation // 2)
            slot.release(generation // 2)


if __name__ == "__main__":
    unittest.main()
