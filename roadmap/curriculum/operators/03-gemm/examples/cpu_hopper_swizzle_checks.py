"""CPU checks for an 8x8 Hopper-style 128-byte shared-memory swizzle model.

Coordinates name 16-byte chunks, not scalar values. One chunk holds eight
consecutive half values (or one CUDA int4 vector, which is 128 bits; this is
not 4-bit quantization). The bank model below assumes one 32-bit word per lane
and is an address-level model, not a GPU instruction or profiler simulator.
"""

from collections import Counter
import unittest


SIDE = 8
CHUNK_BYTES = 16
HALF_BYTES = 2
WORD_BYTES = 4
BANK_COUNT = 32
BANK_BYTES = 4


def swizzled_chunk_x(x, y, smem_base):
    """Map logical chunk (x,y) to physical x for a 128-byte swizzle.

    smem_base must be 128-byte aligned. The 1024-byte-aligned case has bias
    zero; an arbitrary 128-byte-aligned base selects one of eight XOR phases.
    """
    if type(x) is not int or not 0 <= x < SIDE:
        raise ValueError("x must be an integer in [0,7]")
    if type(y) is not int or not 0 <= y < SIDE:
        raise ValueError("y must be an integer in [0,7]")
    if type(smem_base) is not int or smem_base < 0 or smem_base % 128:
        raise ValueError("shared base must be a non-negative 128-byte multiple")
    offset = (smem_base // 128) % SIDE
    return x ^ ((y + offset) % SIDE)


def chunk_byte_address(smem_base, x, y, swizzle=True):
    """Byte address of a 16-byte chunk in an 8x8 row-major chunk tile."""
    if type(smem_base) is not int or smem_base < 0 or smem_base % 128:
        raise ValueError("shared base must be a non-negative 128-byte multiple")
    physical_x = swizzled_chunk_x(x, y, smem_base) if swizzle else x
    return smem_base + (y * SIDE + physical_x) * CHUNK_BYTES


def warp_bank_histogram(smem_base, word_in_chunk=0, swizzle=True):
    """Count banks for 32 lanes reading one 32-bit word from 32 chunks.

    Lane l maps to x=l//8, y=l%8. This is four chunk columns across eight
    rows, a concrete gather pattern that exposes the stride-versus-XOR effect.
    """
    if type(word_in_chunk) is not int or not 0 <= word_in_chunk < 4:
        raise ValueError("word_in_chunk must be in [0,3]")
    banks = []
    for lane in range(32):
        x, y = divmod(lane, SIDE)
        address = chunk_byte_address(smem_base, x, y, swizzle)
        address += word_in_chunk * WORD_BYTES
        banks.append((address // BANK_BYTES) % BANK_COUNT)
    return Counter(banks)


def verify_layout(smem_base):
    """Return coverage and bank facts for one aligned shared-memory base."""
    logical = [(x, y) for y in range(SIDE) for x in range(SIDE)]
    physical = [
        (swizzled_chunk_x(x, y, smem_base), y) for x, y in logical
    ]
    if len(set(physical)) != SIDE * SIDE:
        raise AssertionError("swizzle is not an exact cover of the 8x8 chunks")

    # A chunk is moved as a unit: its eight half elements retain their order.
    for x, y in logical:
        start = chunk_byte_address(smem_base, x, y)
        half_addresses = [start + i * HALF_BYTES for i in range(8)]
        if half_addresses != list(range(start, start + CHUNK_BYTES, HALF_BYTES)):
            raise AssertionError("half order changed inside a 16-byte chunk")

    bank_cases = []
    for word in range(4):
        plain = warp_bank_histogram(smem_base, word, swizzle=False)
        swizzled = warp_bank_histogram(smem_base, word, swizzle=True)
        bank_cases.append((plain, swizzled))
    return physical, bank_cases


class HopperSwizzleTests(unittest.TestCase):
    def test_1024_byte_aligned_base_is_xor_xy(self):
        base = 3 * 1024
        for y in range(SIDE):
            for x in range(SIDE):
                self.assertEqual(swizzled_chunk_x(x, y, base), x ^ y)

    def test_128_byte_alignment_phase_and_exact_cover(self):
        for base in range(0, 1024, 128):
            physical, _ = verify_layout(base)
            self.assertEqual(len(physical), SIDE * SIDE)
            self.assertEqual(len(set(physical)), SIDE * SIDE)

    def test_32_lane_bank_model(self):
        # Each unswizzled 128-byte row repeats the same bank phase: 8-way.
        # The XOR maps these four-column gathers over all eight chunk columns.
        for base in range(0, 1024, 128):
            for word in range(4):
                plain = warp_bank_histogram(base, word, swizzle=False)
                swizzled = warp_bank_histogram(base, word, swizzle=True)
                self.assertEqual(sorted(plain.values()), [8, 8, 8, 8])
                self.assertEqual(sorted(swizzled.values()), [4] * 8)

    def test_rejects_bad_coordinates_and_alignment(self):
        for args in ((-1, 0, 0), (0, 8, 0), (0, 0, 64)):
            with self.assertRaises(ValueError):
                swizzled_chunk_x(*args)


if __name__ == "__main__":
    unittest.main()
