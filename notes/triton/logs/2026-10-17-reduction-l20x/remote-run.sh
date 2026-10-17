#!/usr/bin/env bash
set -u
set -o pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TASK_ID="reduction-check-20261017-01a01efb"
PERSONAL_ROOT="/home/LiuYan/w00939120/aiinffra-test"
PYTHON="$PERSONAL_ROOT/envs/torch-2.13.0-cu126/bin/python"
GPU_UUID="GPU-b2d439c0-5ad1-b1f4-b639-28da8cd5bf6a"
EXIT_FILE="$TASK_DIR/exit.txt"

finish() {
    status=$?
    printf '%s\n' "$status" > "$EXIT_FILE"
    {
        printf 'task_id=%s\nexit_code=%s\nfinished_utc=' "$TASK_ID" "$status"
        date -u +%Y-%m-%dT%H:%M:%SZ
        printf '\n'
        nvidia-smi --query-gpu=uuid,name,memory.used,memory.total --format=csv,noheader || true
        printf '\ncompute-apps-after:\n'
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
        printf '\nreduction-python-processes-after:\n'
        ps -eo pid=,args= | awk -v p="$TASK_DIR/source/reduction_check.py" 'index($0,p) {print}' || true
    } > "$TASK_DIR/after.txt"
    exit "$status"
}
trap finish EXIT

mkdir -p "$TASK_DIR/tmp" "$TASK_DIR/cache" "$TASK_DIR/triton-cache" "$TASK_DIR/cuda-cache"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export TMPDIR="$TASK_DIR/tmp" TEMP="$TASK_DIR/tmp" TMP="$TASK_DIR/tmp"
export XDG_CACHE_HOME="$TASK_DIR/cache"
export TRITON_CACHE_DIR="$TASK_DIR/triton-cache"
export CUDA_CACHE_PATH="$TASK_DIR/cuda-cache"
export TORCHINDUCTOR_CACHE_DIR="$TASK_DIR/cache/torchinductor"

{
    printf 'task_id=%s\n' "$TASK_ID"
    printf 'task_dir=%s\n' "$TASK_DIR"
    printf 'personal_root=%s\n' "$PERSONAL_ROOT"
    printf 'identity='; id
    printf 'hostname='; hostname
    printf 'container_cgroup='; tr '\n' ' ' < /proc/1/cgroup; printf '\n'
    printf 'python_path=%s\n' "$PYTHON"
    "$PYTHON" --version
    "$PYTHON" -c 'import torch,triton; print("torch="+torch.__version__); print("triton="+triton.__version__); print("torch_cuda="+str(torch.version.cuda))'
    printf '\nsource_hashes:\n'
    sha256sum "$TASK_DIR"/source/*.py
    printf '\ngpu_before:\n'
    nvidia-smi --query-gpu=uuid,name,memory.used,memory.total --format=csv,noheader
    printf '\ncompute_apps_before:\n'
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} > "$TASK_DIR/environment.txt" 2>&1

"$PYTHON" -c 'import ast,pathlib,sys; paths=sys.argv[1:]; [ast.parse(pathlib.Path(p).read_text(encoding="utf-8"), filename=p) for p in paths]; print("AST parse passed:", *paths)' \
    "$TASK_DIR/source/reduction_single_block.py" \
    "$TASK_DIR/source/reduction_multi_block.py" \
    "$TASK_DIR/source/reduction_check.py" > "$TASK_DIR/offline.log" 2>&1 || exit $?
"$PYTHON" "$TASK_DIR/source/reduction_check.py" --cpu-check >> "$TASK_DIR/offline.log" 2>&1 || exit $?

# Recheck the selected UUID and process table immediately before CUDA execution.
nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader > "$TASK_DIR/gpu-launch-check.txt" || exit $?
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > "$TASK_DIR/compute-apps-launch-check.txt" || exit $?
awk -F, -v uuid="$GPU_UUID" 'BEGIN{found=0} {gsub(/[[:space:]]/,"",$1); if($1==uuid){found=1; gsub(/[[:space:]]/,"",$2); if($2!="0MiB") exit 2}} END{if(!found) exit 3}' "$TASK_DIR/gpu-launch-check.txt" || exit $?
if grep -F "$GPU_UUID" "$TASK_DIR/compute-apps-launch-check.txt" >/dev/null; then
    printf 'Refusing launch: selected UUID has an existing compute-app process\n' >> "$TASK_DIR/offline.log"
    exit 70
fi

timeout --signal=TERM 120s "$PYTHON" "$TASK_DIR/source/reduction_check.py" --correctness-only \
    > "$TASK_DIR/gpu-correctness.log" 2>&1
