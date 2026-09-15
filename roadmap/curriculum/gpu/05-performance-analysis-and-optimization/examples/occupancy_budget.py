#!/usr/bin/env python3
"""Hand-calculate a simplified SM residency budget.

The numbers are deliberately explicit teaching assumptions. They are not a
replacement for compiler allocation reports or the CUDA occupancy API.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assumptions:
    sm_registers_32bit: int = 65_536
    sm_shared_bytes: int = 96 * 1024
    sm_threads: int = 2_048
    sm_warps: int = 64
    sm_blocks: int = 32
    warp_size: int = 32
    block_threads: int = 256


def budget(regs_per_thread: int, shared_bytes_per_block: int,
           a: Assumptions = Assumptions()) -> dict[str, int | float]:
    warps_per_block = (a.block_threads + a.warp_size - 1) // a.warp_size
    by_threads = a.sm_threads // a.block_threads
    by_warps = a.sm_warps // warps_per_block
    by_registers = a.sm_registers_32bit // (a.block_threads * regs_per_thread)
    by_shared = a.sm_shared_bytes // shared_bytes_per_block
    resident = min(a.sm_blocks, by_threads, by_warps, by_registers, by_shared)
    active_warps = resident * warps_per_block
    return {
        "threads": by_threads,
        "warps": by_warps,
        "registers": by_registers,
        "shared": by_shared,
        "blocks": a.sm_blocks,
        "resident_blocks": resident,
        "active_warps": active_warps,
        "occupancy": active_warps / a.sm_warps,
    }


def main() -> None:
    assumptions = Assumptions()
    print("assumptions:", assumptions)
    print("allocation note: the arithmetic below uses logical FP32 registers and bytes;")
    print("compiler register/shared-memory allocation is rounded by architecture-specific granularity.")
    for regs, shared in ((64, 32 * 1024), (96, 48 * 1024)):
        result = budget(regs, shared, assumptions)
        print(f"\nregs/thread={regs} shared/block={shared // 1024} KiB")
        print("  thread limit :", result["threads"], "blocks")
        print("  warp limit   :", result["warps"], "blocks")
        print("  register limit:", result["registers"], "blocks")
        print("  shared limit :", result["shared"], "blocks")
        print("  block limit  :", result["blocks"], "blocks")
        print("  resident     :", result["resident_blocks"], "blocks")
        print("  active warps :", result["active_warps"])
        print("  occupancy    :", f'{result["occupancy"]:.1%}')


if __name__ == "__main__":
    main()
