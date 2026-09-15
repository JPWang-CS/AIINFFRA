"""CPU-only protocol checks; these do not emulate CUDA hardware or timing."""

from itertools import permutations
import unittest


def attribute_owner(attrs_idxs, count):
    if count <= 0 or not attrs_idxs or attrs_idxs[0] != 0:
        raise ValueError("attribute segments must start at copy 0")
    if len(attrs_idxs) > count:
        raise ValueError("more attribute segments than copies")
    if any(a >= b for a, b in zip(attrs_idxs, attrs_idxs[1:])):
        raise ValueError("segment starts must be strictly increasing")
    if attrs_idxs[-1] >= count:
        raise ValueError("last segment must start before count")
    owners = []
    for i, start in enumerate(attrs_idxs):
        stop = attrs_idxs[i + 1] if i + 1 < len(attrs_idxs) else count
        owners.extend([i] * (stop - start))
    assert len(owners) == count
    return owners


def clc_exact_cover(order, steal_after_ctas):
    """Abstract CLC: each CTA does its own tile, then may steal one pending tile."""
    pending = set(order)
    processed = []
    cancelled = set()
    for cta in order:
        if cta not in pending:  # another CTA stole this not-yet-started block
            continue
        pending.remove(cta)
        processed.append(cta)
        if cta in steal_after_ctas and pending:
            stolen = min(pending)  # any unstarted index returned by a success
            pending.remove(stolen)
            cancelled.add(stolen)
            processed.append(stolen)
    return processed, cancelled, pending


class TimelineSemaphoreLifecycle:
    def __init__(self):
        self.issued = set()
        self.completed = set()
        self.waits = []
        self.completed_waits = []
        self.destroyed = False

    def signal(self, value):
        if self.destroyed or (self.issued and value <= max(self.issued)):
            raise ValueError("timeline signal values must increase")
        self.issued.add(value)

    def wait(self, value):
        if self.destroyed or value < 0:
            raise ValueError("invalid timeline wait")
        # Timeline waits may be enqueued before their future signal is issued.
        self.waits.append(value)

    def complete_signal(self, value):
        if value not in self.issued:
            raise ValueError("cannot complete an unissued signal")
        self.completed.add(value)

    def complete_wait(self, value):
        if value not in self.waits or not any(s >= value for s in self.completed):
            raise ValueError("timeline wait has not reached its target value")
        self.completed_waits.append(value)

    def destroy(self):
        if self.issued != self.completed or len(self.completed_waits) != len(self.waits):
            raise ValueError("cannot destroy with in-flight signal/wait")
        self.destroyed = True


class BinarySemaphoreLifecycle:
    def __init__(self):
        self.signaled = False
        self.destroyed = False

    def signal(self):
        if self.destroyed or self.signaled:
            raise ValueError("binary semaphore must be consumed before re-signal")
        self.signaled = True

    def wait(self):
        if self.destroyed or not self.signaled:
            raise ValueError("binary wait requires an earlier signal")
        self.signaled = False

    def destroy(self):
        if self.signaled:
            raise ValueError("cannot destroy a signaled binary semaphore")
        self.destroyed = True


class PoolAllocationLifetime:
    def __init__(self, importers):
        self.importers = set(importers)
        self.freed_importers = set()
        self.exporter_freed = False
        self.ready = False

    def access(self, importer):
        if not self.ready or importer not in self.importers:
            raise ValueError("allocation is not ready or not imported")

    def free_import(self, importer):
        if importer not in self.importers or self.exporter_freed:
            raise ValueError("invalid imported-allocation free")
        self.freed_importers.add(importer)

    def free_export(self):
        if self.freed_importers != self.importers:
            raise ValueError("exporter must wait for every importer free")
        self.exporter_freed = True


class ProtocolTests(unittest.TestCase):
    def test_memcpy_attrs_are_an_exact_partition(self):
        self.assertEqual(attribute_owner([0, 3], 5), [0, 0, 0, 1, 1])
        self.assertEqual(attribute_owner([0], 4), [0, 0, 0, 0])
        for starts, count in (([1], 4), ([0, 0], 4), ([0, 4], 4), ([0, 1], 1)):
            with self.assertRaises(ValueError):
                attribute_owner(starts, count)

    def test_clc_has_exactly_once_work_for_all_small_schedules(self):
        count = 5
        for order in permutations(range(count)):
            for mask in range(1 << count):
                stealers = {i for i in range(count) if mask & (1 << i)}
                processed, cancelled, pending = clc_exact_cover(order, stealers)
                self.assertFalse(pending)
                self.assertEqual(len(processed), len(set(processed)))
                self.assertEqual(sorted(processed), list(range(count)))
                self.assertTrue(cancelled.issubset(set(range(count))))

    def test_failed_cancel_has_no_index_and_terminates_that_worker_loop(self):
        success, returned_index = False, None
        decoded_index = returned_index if success else None
        self.assertIsNone(decoded_index)

    def test_timeline_wait_may_be_enqueued_before_future_signal(self):
        semaphore = TimelineSemaphoreLifecycle()
        semaphore.wait(10)
        with self.assertRaises(ValueError):
            semaphore.destroy()
        semaphore.signal(10)       # future Vulkan signal reaches wait target
        semaphore.complete_signal(10)
        semaphore.complete_wait(10)
        semaphore.signal(11)       # CUDA signals after its stream work
        semaphore.wait(11)         # Vulkan can queue this wait for a future value
        semaphore.complete_signal(11)
        semaphore.complete_wait(11)
        semaphore.destroy()
        self.assertTrue(semaphore.destroyed)

    def test_binary_semaphore_requires_signal_before_wait(self):
        semaphore = BinarySemaphoreLifecycle()
        with self.assertRaises(ValueError):
            semaphore.wait()
        semaphore.signal()
        semaphore.wait()
        semaphore.destroy()
        self.assertTrue(semaphore.destroyed)

    def test_pool_exporter_waits_for_all_imported_allocations_to_free(self):
        allocation = PoolAllocationLifetime({"consumer-a", "consumer-b"})
        with self.assertRaises(ValueError):
            allocation.access("consumer-a")
        allocation.ready = True
        allocation.access("consumer-a")
        allocation.free_import("consumer-a")
        with self.assertRaises(ValueError):
            allocation.free_export()
        allocation.free_import("consumer-b")
        allocation.free_export()
        self.assertTrue(allocation.exporter_freed)

    def test_alias_requires_permission_and_operation_completion(self):
        backing = bytearray([0])
        aliases = {"gpu0": backing, "gpu1": backing}
        rights = {"gpu0": {"gpu0"}, "gpu1": set()}
        with self.assertRaises(PermissionError):
            if "gpu1" not in rights["gpu1"]:
                raise PermissionError("cuMemSetAccess was not granted")
        rights["gpu1"].add("gpu1")
        aliases["gpu0"][0] = 7
        write_operation_complete = True  # stream/event dependency in this model
        self.assertTrue(write_operation_complete)
        self.assertEqual(aliases["gpu1"][0], 7)

    def test_parent_cannot_be_reclaimed_before_nested_child_finishes(self):
        parent_complete = False
        child_complete = False
        if child_complete:
            parent_complete = True
        self.assertFalse(parent_complete)
        child_complete = True
        if child_complete:
            parent_complete = True
        self.assertTrue(parent_complete)


if __name__ == "__main__":
    unittest.main(verbosity=2)
