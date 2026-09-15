#!/usr/bin/env python3
"""Show explicit CUDA Driver bindings, context, stream, and device-pointer lifetime."""

import sys

import numpy as np


def main() -> int:
    try:
        from cuda.bindings import driver as cu
    except ImportError as error:
        print(f"SKIP: cuda.bindings is unavailable: {error}")
        return 77

    success = cu.CUresult.CUDA_SUCCESS

    def checked(call, where):
        result = call()
        status, *values = result
        if status != success:
            _, name = cu.cuGetErrorName(status)
            _, message = cu.cuGetErrorString(status)
            raise RuntimeError(f"{where}: {name}: {message}")
        return values[0] if values else None

    def cleanup(call, where):
        try:
            checked(call, where)
        except Exception as error:
            print(f"cleanup warning: {error}", file=sys.stderr)

    checked(lambda: cu.cuInit(0), "cuInit")
    previous_context = checked(cu.cuCtxGetCurrent, "cuCtxGetCurrent")
    driver_version = checked(cu.cuDriverGetVersion, "cuDriverGetVersion")
    device_count = checked(cu.cuDeviceGetCount, "cuDeviceGetCount")
    print(f"driver API version={driver_version}; visible devices={device_count}")
    if device_count == 0:
        print("SKIP: no CUDA device is visible")
        return 77

    device = checked(lambda: cu.cuDeviceGet(0), "cuDeviceGet")
    context = None
    stream = None
    device_ptr = None
    context_is_current = False
    host_word = np.full(1, 0xA5A5A5A5, dtype=np.uint32)
    try:
        context = checked(lambda: cu.cuDevicePrimaryCtxRetain(device),
                          "cuDevicePrimaryCtxRetain")
        checked(lambda: cu.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        context_is_current = True
        stream = checked(lambda: cu.cuStreamCreate(0), "cuStreamCreate")
        device_ptr = checked(lambda: cu.cuMemAlloc(4), "cuMemAlloc")
        print(f"device pointer handle={device_ptr}; allocated bytes=4")

        # The 32-bit pattern is written asynchronously in this stream.
        checked(lambda: cu.cuMemsetD32Async(device_ptr, 0, 1, stream),
                "cuMemsetD32Async")
        checked(lambda: cu.cuStreamSynchronize(stream), "cuStreamSynchronize")
        checked(lambda: cu.cuMemcpyDtoH(host_word, device_ptr, host_word.nbytes),
                "cuMemcpyDtoH")
        np.testing.assert_array_equal(host_word, np.array([0], dtype=np.uint32))
        print(f"D2H verification passed: {host_word[0]}")
    finally:
        # Never release storage while work queued in its stream can still use it.
        if stream is not None:
            cleanup(lambda: cu.cuStreamSynchronize(stream), "final stream sync")
        if device_ptr is not None:
            cleanup(lambda: cu.cuMemFree(device_ptr), "cuMemFree")
        if stream is not None:
            cleanup(lambda: cu.cuStreamDestroy(stream), "cuStreamDestroy")
        if context_is_current:
            cleanup(lambda: cu.cuCtxSetCurrent(previous_context),
                    "restore previous current context")
        if context is not None:
            cleanup(lambda: cu.cuDevicePrimaryCtxRelease(device),
                    "cuDevicePrimaryCtxRelease")
    return 0


if __name__ == "__main__":
    sys.exit(main())
