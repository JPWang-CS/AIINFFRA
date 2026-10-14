#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
set +o history

readonly REPOSITORY_URL="https://github.com/JPWang-CS/AIINFFRA.git"
workspace_arg=""
allowed_arg=""
expected_sha=""

usage() {
  printf 'Usage: %s --workspace-root ABS_PATH --allowed-root ABS_PATH --expected-sha 40_LOWERCASE_HEX\n' "$0" >&2
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --workspace-root) (($# >= 2)) || { usage; exit 2; }; workspace_arg="$2"; shift 2 ;;
    --allowed-root) (($# >= 2)) || { usage; exit 2; }; allowed_arg="$2"; shift 2 ;;
    --expected-sha) (($# >= 2)) || { usage; exit 2; }; expected_sha="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown argument: $1" ;;
  esac
done

[[ -n "$workspace_arg" && -n "$allowed_arg" && -n "$expected_sha" ]] || { usage; exit 2; }
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || die "expected SHA must be 40 lowercase hexadecimal characters"
readonly EXPECTED_SHA="$expected_sha"
[[ "$workspace_arg" == /* && "$allowed_arg" == /* ]] || die "both roots must be absolute paths"
[[ ! -L "$allowed_arg" ]] || die "allowed root is a symlink"
allowed_root="$(realpath -e -- "$allowed_arg")"
[[ -d "$allowed_root" ]] || die "allowed root is not a directory"
case "$allowed_root" in /|/home|/root|/tmp|/var|/mnt) die "allowed root is too broad" ;; esac

workspace_parent="$(realpath -e -- "$(dirname -- "$workspace_arg")")"
[[ "$workspace_parent" == "$allowed_root" ]] || die "workspace must be a direct child of the allowed root"
workspace_root="$workspace_parent/$(basename -- "$workspace_arg")"
[[ "$workspace_root" != "$allowed_root" && ! -L "$workspace_root" ]] || die "invalid or symlinked workspace root"
if [[ -e "$workspace_root" ]]; then
  [[ -d "$workspace_root" ]] || die "workspace root exists and is not a directory"
  [[ "$(realpath -e -- "$workspace_root")" == "$workspace_root" ]] || die "workspace root resolves outside its requested path"
else
  mkdir -m 700 -- "$workspace_root"
fi

task_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_root="$workspace_root/runs"
[[ ! -L "$run_root" ]] || die "runs directory is a symlink"
if [[ -e "$run_root" ]]; then [[ -d "$run_root" ]] || die "runs path is not a directory"; else mkdir -m 700 -- "$run_root"; fi
task_dir="$run_root/$task_id"
mkdir -m 700 -- "$task_dir"
log_file="$task_dir/setup.log"
exit_file="$task_dir/exit.txt"
printf '%s\n' "$task_id" > "$task_dir/task-id.txt"
exec > >(tee -a "$log_file") 2>&1

finish() {
  local rc=$?
  trap - EXIT
  set +e
  printf 'task_id=%s\nexit_code=%s\ncompleted_utc=%s\n' "$task_id" "$rc" "$(date -u +%FT%TZ)" > "$exit_file"
  local receipt_rc=$?
  if ((receipt_rc != 0 && rc == 0)); then rc=$receipt_rc; fi
  exit "$rc"
}
trap finish EXIT

printf 'task_id=%s\nworkspace_root=%s\n' "$task_id" "$workspace_root"
[[ ! -L "$workspace_root/AIINFFRA" && ! -e "$workspace_root/AIINFFRA" ]] || die "repository target already exists"
[[ ! -L "$workspace_root/envs" ]] || die "envs directory is a symlink"
if [[ -e "$workspace_root/envs" ]]; then [[ -d "$workspace_root/envs" ]] || die "envs path is not a directory"; else mkdir -m 700 -- "$workspace_root/envs"; fi
env_path="$workspace_root/envs/torch-2.7.1-cu128"
[[ ! -L "$env_path" && ! -e "$env_path" ]] || die "venv target already exists"
[[ -z "$(find "$workspace_root/envs" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "envs directory is not empty"

for name in AIINFFRA envs runs cache tmp; do
  if [[ -e "$workspace_root/$name" && ! -d "$workspace_root/$name" ]]; then die "$name exists and is not a directory"; fi
  [[ ! -L "$workspace_root/$name" ]] || die "$name is a symlink"
done
while IFS= read -r entry; do
  case "$(basename -- "$entry")" in AIINFFRA|envs|runs|cache|tmp) ;; *) die "unexpected entry under workspace root" ;; esac
done < <(find "$workspace_root" -mindepth 1 -maxdepth 1 -print)
for name in cache tmp; do [[ -e "$workspace_root/$name" ]] || mkdir -m 700 -- "$workspace_root/$name"; done
cache_root="$workspace_root/cache/$task_id"
tmp_root="$workspace_root/tmp/$task_id"
mkdir -m 700 -- "$cache_root" "$tmp_root"

for name in ${!PIP_@}; do unset "$name"; done
export PIP_CONFIG_FILE=/dev/null
export PIP_CACHE_DIR="$cache_root/pip"
export TMPDIR="$tmp_root"
export XDG_CACHE_HOME="$cache_root/xdg"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCH_HOME="$cache_root/torch"
export CUDA_CACHE_PATH="$cache_root/cuda"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1
export HISTFILE=/dev/null HISTSIZE=0 HISTFILESIZE=0
export GIT_TERMINAL_PROMPT=0 GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
mkdir -m 700 -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"

capture_toolchain() {
  local prefix="$1"
  nvidia-smi --query-gpu=driver_version --format=csv,noheader > "$task_dir/$prefix-driver.txt"
  nvcc --version > "$task_dir/$prefix-nvcc.txt" 2>&1
  gcc --version > "$task_dir/$prefix-gcc.txt" 2>&1
  g++ --version > "$task_dir/$prefix-gxx.txt" 2>&1
  python3 --version > "$task_dir/$prefix-python.txt" 2>&1
}
capture_toolchain before
python3 --version | grep -Fx 'Python 3.12.3' >/dev/null || die "system python3 is not 3.12.3"

remote_sha="$(git -c credential.helper= -c core.askPass= ls-remote "$REPOSITORY_URL" refs/heads/main | awk 'NR==1 {print $1}')"
[[ "$remote_sha" == "$EXPECTED_SHA" ]] || die "origin/main is not the approved SHA; clone stopped"
git -c credential.helper= -c core.askPass= clone --branch main --single-branch -- "$REPOSITORY_URL" "$workspace_root/AIINFFRA"
repo="$workspace_root/AIINFFRA"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$EXPECTED_SHA" ]] || die "cloned HEAD does not match approved SHA"
[[ "$(git -C "$repo" branch --show-current)" == main ]] || die "clone is not on main"
[[ "$(git -C "$repo" rev-parse --abbrev-ref --symbolic-full-name '@{u}')" == origin/main ]] || die "main does not track origin/main"

python3 -m venv "$env_path"
venv_python="$env_path/bin/python"
"$venv_python" -c 'import sys; assert sys.prefix != sys.base_prefix, "not a virtual environment"'
"$venv_python" -m pip --version | tee "$task_dir/pip-version.txt"
"$venv_python" -m pip --disable-pip-version-check install --no-input --report "$task_dir/pip-torch-report.json" --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1+cu128'
"$venv_python" -m pip --disable-pip-version-check install --no-input --report "$task_dir/pip-pypi-report.json" --index-url https://pypi.org/simple 'triton==3.3.1' 'numpy==1.26.4'
"$venv_python" -m pip check | tee "$task_dir/pip-check.txt"
"$venv_python" -m pip freeze > "$task_dir/pip-freeze.txt"
PYTORCH_NVML_BASED_CUDA_CHECK=1 "$venv_python" - <<'PY' | tee "$task_dir/python-smoke.txt"
import sys
import numpy
import torch
import triton

assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert triton.__version__ == "3.3.1", triton.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
assert sys.prefix != sys.base_prefix, "Python is not running in a venv"
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("numpy:", numpy.__version__)
cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()
assert cuda_available, "CUDA is not available"
assert device_count > 0, "no CUDA devices are visible"
print("cuda_available:", cuda_available)
print("device_count:", device_count)
print("arch_list:", torch.cuda.get_arch_list())
PY
capture_toolchain after
for tool in driver nvcc gcc gxx python; do
  cmp -s "$task_dir/before-$tool.txt" "$task_dir/after-$tool.txt" || die "$tool version output changed during setup"
done
printf 'setup complete; no course code or GPU tensors were run\n'
