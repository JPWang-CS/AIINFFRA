#!/usr/bin/env bash
set -eu

# Read-only tool inventory. It does not change driver state, clocks, files,
# persistence settings, or profiler permissions.
printf '%s\n' '== host =='
uname -a || true
printf '%s\n' '== compiler and profilers =='
for tool in nvcc nsys ncu compute-sanitizer cuobjdump nvdisasm; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%s: ' "$tool"
    "$tool" --version 2>&1 | head -n 1 || true
  else
    printf '%s: unavailable\n' "$tool"
  fi
done
printf '%s\n' '== visible devices =='
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,driver_version \
    --format=csv,noheader || true
else
  printf '%s\n' 'nvidia-smi: unavailable'
fi
printf '%s\n' '== environment =='
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
printf 'CUDA_LAUNCH_BLOCKING=%s\n' "${CUDA_LAUNCH_BLOCKING-<unset>}"
