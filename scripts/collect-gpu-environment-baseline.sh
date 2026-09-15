#!/usr/bin/env bash

# Read-only GPU environment collector. Run from the AIINFFRA repository root.
set -u

stamp="$(date +%Y%m%d-%H%M%S)"
out_dir="${1:-results/gpu-environment/${stamp}}"
mkdir -p "${out_dir}"

run_to_file() {
  local file="$1"
  shift
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } >>"${out_dir}/${file}" 2>&1 || true
}

{
  echo "GPU environment baseline collection"
  echo "timestamp=${stamp}"
  echo "cwd=$(pwd)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
  echo "output=${out_dir}"
} >"${out_dir}/00-summary.txt"

run_to_file 01-system.txt date -Is
run_to_file 01-system.txt uname -a
if command -v lscpu >/dev/null 2>&1; then run_to_file 01-system.txt lscpu; fi
if command -v numactl >/dev/null 2>&1; then run_to_file 01-system.txt numactl --hardware; fi
if [[ -r /etc/os-release ]]; then run_to_file 01-system.txt cat /etc/os-release; fi

if command -v nvidia-smi >/dev/null 2>&1; then
  run_to_file 02-nvidia-smi.txt nvidia-smi
  run_to_file 02-nvidia-smi.txt nvidia-smi -q
  nvidia-smi --query-gpu=timestamp,index,name,uuid,pci.bus_id,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,memory.total,memory.used,utilization.gpu,utilization.memory --format=csv >"${out_dir}/03-nvidia-smi-query.csv" 2>&1 || true
  nvidia-smi topo -m >"${out_dir}/04-topology.txt" 2>&1 || true
else
  echo "nvidia-smi unavailable" >"${out_dir}/02-nvidia-smi.txt"
fi

{
  command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "nvcc unavailable"
  command -v python >/dev/null 2>&1 && python --version || echo "python unavailable"
  command -v python3 >/dev/null 2>&1 && python3 --version || true
} >"${out_dir}/05-toolchain.txt" 2>&1

python_cmd=""
if command -v python >/dev/null 2>&1; then python_cmd="python"; elif command -v python3 >/dev/null 2>&1; then python_cmd="python3"; fi

if [[ -n "${python_cmd}" ]]; then
  "${python_cmd}" - <<'PY' >"${out_dir}/06-python-stack.txt" 2>&1 || true
import platform

print("python:", platform.python_version())
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("visible device count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"device[{index}].name:", props.name)
        print(f"device[{index}].cc:", f"{props.major}.{props.minor}")
        print(f"device[{index}].sm_count:", props.multi_processor_count)
        print(f"device[{index}].total_memory:", props.total_memory)
        print(f"device[{index}].warp_size:", props.warp_size)
except Exception as exc:
    print("torch probe failed:", repr(exc))
try:
    import triton
    print("triton:", triton.__version__)
except Exception as exc:
    print("triton probe failed:", repr(exc))
PY
else
  echo "python unavailable" >"${out_dir}/06-python-stack.txt"
fi

echo "GPU environment collection complete: ${out_dir}"
echo "Next: run CUDA samples deviceQuery and bandwidthTest, then complete the first-chapter acceptance record."
