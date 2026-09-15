"""Scalar reference models for quantization; no GPU or framework dependency."""
import math


def quantize_affine(value, scale, zero_point=0, qmin=-128, qmax=127):
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("finite input and positive finite scale required")
    if any(not isinstance(x,int) for x in (zero_point,qmin,qmax)) or qmin > qmax or not qmin <= zero_point <= qmax:
        raise ValueError("invalid quantized range or zero point")
    scaled = min(qmax-zero_point, max(qmin-zero_point, value / scale))
    rounded = round(scaled) + zero_point
    return min(qmax, max(qmin, rounded))


def pack_int4(values):
    if any(not isinstance(x, int) or not -8 <= x <= 7 for x in values):
        raise ValueError("signed INT4 values must lie in [-8,7]")
    packed = []
    for i in range(0, len(values), 2):
        low = values[i] & 15
        high = (values[i + 1] & 15) if i + 1 < len(values) else 0
        packed.append(low | (high << 4))
    return packed


def unpack_int4(packed, count):
    if count < 0 or len(packed) != (count + 1) // 2:
        raise ValueError("packed size does not match logical count")
    if any(not isinstance(x, int) or not 0 <= x <= 255 for x in packed):
        raise ValueError("byte outside [0,255]")
    output = []
    for i in range(count):
        nibble = (packed[i // 2] >> (4 * (i % 2))) & 15
        output.append(nibble - 16 if nibble >= 8 else nibble)
    return output


def corrected_dot(a, b, za, zb):
    if len(a) != len(b):
        raise ValueError("reduction dimensions differ")
    return sum(x*y for x, y in zip(a, b)) - zb*sum(a) - za*sum(b) + len(a)*za*zb


def block_scale(x, scales, tile):
    if tile <= 0 or not x or not x[0]:
        raise ValueError("nonempty matrix and positive tile required")
    m, n = len(x), len(x[0])
    if any(len(row) != n for row in x):
        raise ValueError("ragged matrix")
    if len(scales) != (m+tile-1)//tile or any(len(row) != (n+tile-1)//tile for row in scales):
        raise ValueError("wrong scale matrix")
    return [[x[r][c] * scales[r//tile][c//tile] for c in range(n)] for r in range(m)]


if __name__ == "__main__":
    assert [quantize_affine(x, 0.5) for x in [0.25, 0.75, -0.25, -0.75]] == [0, 2, 0, -2]
    assert quantize_affine(1000, 1) == 127
    assert quantize_affine(-1000, 1) == -128
    assert quantize_affine(0, 0.25, 5) == 5
    assert quantize_affine(1.0, 1e-320) == 127
    for a in range(-8, 8):
        for b in range(-8, 8):
            assert unpack_int4(pack_int4([a, b]), 2) == [a, b]
    for n in [0, 1, 3, 16, 31, 129]:
        values = [i % 16 - 8 for i in range(n)]
        assert unpack_int4(pack_int4(values), n) == values
    assert pack_int4([-1, 2, -8]) == [47, 8]
    a, b, za, zb = [1, -4, 7], [5, 2, -3], 2, -1
    assert corrected_dot(a, b, za, zb) == sum((x-za)*(y-zb) for x,y in zip(a,b))
    assert block_scale([[1,2,3],[4,5,6],[7,8,9]], [[2,3],[4,5]], 2) == [[2,4,9],[8,10,18],[28,32,45]]
    assert 128 * 4 / 8 + 2 == 66  # one FP16 scale per 128 signed INT4 values
    assert 16 * 4 / 8 + 1 == 9    # E2M1 payload plus one 8-bit block scale
    x,w,s = [3,2],[1,4],[2,.5]
    assert sum(a*b for a,b in zip(x,w)) == sum((a/c)*(c*b) for a,b,c in zip(x,w,s))
    quadratic = lambda a,b: 2*a*a+2*a*b+2*b*b
    assert math.isclose(quadratic(.4,0),.32)
    assert math.isclose(quadratic(.4,-.2),.24)
    for scale in [0, -1, float("nan"), float("inf")]:
        try:
            quantize_affine(1, scale)
        except ValueError:
            continue
        raise AssertionError("invalid scale accepted")
    print("PASS: quantization ties/clipping, all INT4 pairs and tails, zero-point correction, block scale")
