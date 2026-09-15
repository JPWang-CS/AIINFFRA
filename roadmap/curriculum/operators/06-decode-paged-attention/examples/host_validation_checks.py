"""stdlib-only regressions for wrapper alias and append ownership checks."""

from paged_decode_host import (
    byte_ranges_overlap,
    validate_head_counts,
    validate_memory_contract,
    validate_private_append_targets,
    validate_trusted_capacity,
)


class FakeTensor:
    def __init__(self, pointer, nbytes):
        self.pointer, self.nbytes = pointer, nbytes

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1


def main():
    assert byte_ranges_overlap(100, 16, 100, 16)  # catches self-comparison regressions
    assert byte_ranges_overlap(100, 16, 112, 8)
    assert not byte_ranges_overlap(100, 16, 116, 8)
    q, ck, cv = FakeTensor(1000, 64), FakeTensor(2000, 128), FakeTensor(2200, 128)
    lengths, table, out = FakeTensor(3000, 8), FakeTensor(3100, 16), FakeTensor(3200, 64)
    validate_memory_contract(q, ck, cv, lengths, table, None, None, out)  # legal read
    validate_memory_contract(q, ck, cv, lengths, table, FakeTensor(3300, 64), FakeTensor(3400, 64), out)
    try:
        validate_memory_contract(q, ck, cv, lengths, table, None, None, q)
    except ValueError:
        pass
    else:
        raise AssertionError("out alias was accepted")
    try:
        validate_memory_contract(q, ck, cv, lengths, table, ck, FakeTensor(3400, 64), out)
    except ValueError:
        pass
    else:
        raise AssertionError("new/cache alias was accepted")
    validate_private_append_targets([1, 5], [[7, -1], [10, 11]], 12, 4)
    for table in ([[7, -1], [7, 11]], [[12, -1], [10, 11]]):
        try:
            validate_private_append_targets([1, 5], table, 12, 4)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid/shared append target was accepted")
    try:
        validate_head_counts(4, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero Hkv was accepted")
    validate_trusted_capacity(16, 4, 4)
    try:
        validate_trusted_capacity(17, 4, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("trusted over-capacity length was accepted")
    print("PASS: host-only read/append; self/out/cache alias; shared target; zero Hkv; trusted capacity")


if __name__ == "__main__":
    main()
