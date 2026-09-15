#!/usr/bin/env python3
"""A bounded SIMT vector add with explicit dtype, stream, and tail guard."""

import sys

try:
    import numpy as np
    from numba import cuda
except ImportError as error:
    print(f"SKIP: NumPy or Numba-CUDA is unavailable: {error}")
    raise SystemExit(77)


@cuda.jit
def vector_add(a, b, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = a[i] + b[i]


def run_case(n: int) -> None:
    sentinel = np.float32(-12345.25)
    host_a = np.arange(n, dtype=np.float32)
    host_b = np.full(n, 2.0, dtype=np.float32)
    host_out = np.full(n + 1, sentinel, dtype=np.float32)

    if n == 0:
        np.testing.assert_array_equal(host_out, np.array([sentinel], dtype=np.float32))
        print("n=0: host-only empty case; no zero-sized grid was launched")
        return

    threads = 128
    blocks = (n + threads - 1) // threads
    with cuda.gpus[0]:
        stream = cuda.stream()
        device_a = cuda.to_device(host_a, stream=stream)
        device_b = cuda.to_device(host_b, stream=stream)
        device_out = cuda.to_device(host_out, stream=stream)
        vector_add[blocks, threads, stream](device_a, device_b, device_out, n)
        result = device_out.copy_to_host(stream=stream)
        # Keep arrays, stream, and host buffers alive until all queued work ends.
        stream.synchronize()

    expected = np.full(n + 1, sentinel, dtype=np.float32)
    expected[:n] = host_a + host_b
    np.testing.assert_array_equal(result, expected)
    print(f"n={n}: PASS; tail sentinel={result[n]}")


def main() -> int:
    if not cuda.is_available():
        print("SKIP: no usable CUDA device")
        return 77
    for n in (0, 1, 127, 128, 129, 257):
        run_case(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
