#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Source this helper: source scripts/activate-gpu-sandbox.sh\n' >&2
  exit 2
fi

_aiinffra_activate_gpu_sandbox() {
  local helper_path workspace repo env_path cache_root tmp_root session_id path resolved name
  helper_path="$(realpath -e -- "${BASH_SOURCE[0]}")" || { printf 'ERROR: cannot resolve helper path\n' >&2; return 1; }
  workspace="$(realpath -e -- "$(dirname -- "$helper_path")/../..")" || { printf 'ERROR: cannot resolve workspace root\n' >&2; return 1; }
  repo="$(realpath -e -- "$workspace/AIINFFRA")" || { printf 'ERROR: workspace clone is missing\n' >&2; return 1; }
  [[ "$repo" == "$workspace/AIINFFRA" ]] || { printf 'ERROR: helper is outside workspace/AIINFFRA/scripts\n' >&2; return 1; }

  env_path="$workspace/envs/torch-2.7.1-cu128"
  [[ ! -L "$env_path" && -d "$env_path" ]] || { printf 'ERROR: dedicated venv is missing or symlinked\n' >&2; return 1; }
  [[ ! -L "$env_path/bin/activate" && -f "$env_path/bin/activate" ]] || { printf 'ERROR: venv activation file is missing or symlinked\n' >&2; return 1; }
  resolved="$(realpath -e -- "$env_path")" || { printf 'ERROR: cannot resolve venv path\n' >&2; return 1; }
  [[ "$resolved" == "$env_path" ]] || { printf 'ERROR: venv resolves outside workspace\n' >&2; return 1; }

  session_id="${BASHPID:-$$}"
  cache_root="$workspace/cache/runtime"
  tmp_root="$workspace/tmp/session-$session_id"
  local -a dirs=("$workspace/cache" "$cache_root" "$cache_root/pip" "$cache_root/triton" "$cache_root/torch" "$cache_root/cuda" "$cache_root/xdg" "$workspace/tmp" "$tmp_root")
  for path in "${dirs[@]}"; do
    [[ ! -L "$path" ]] || { printf 'ERROR: cache or temp path is a symlink: %s\n' "$path" >&2; return 1; }
    if [[ -e "$path" ]]; then
      [[ -d "$path" ]] || { printf 'ERROR: cache or temp path is not a directory: %s\n' "$path" >&2; return 1; }
    else
      mkdir -m 700 -- "$path" || { printf 'ERROR: cannot create private directory: %s\n' "$path" >&2; return 1; }
    fi
    resolved="$(realpath -e -- "$path")" || { printf 'ERROR: cannot resolve cache or temp path\n' >&2; return 1; }
    case "$resolved" in "$workspace"/*) ;; *) printf 'ERROR: cache or temp path escapes workspace\n' >&2; return 1 ;; esac
  done

  source "$env_path/bin/activate" || { printf 'ERROR: venv activation failed\n' >&2; return 1; }
  export AIINFFRA_WORK_ROOT="$workspace"
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
  unset PYTHONHOME PYTHONPATH
  for name in ${!PIP_@}; do unset "$name"; done
  export PIP_CONFIG_FILE=/dev/null
  export PIP_CACHE_DIR="$cache_root/pip"
  export TRITON_CACHE_DIR="$cache_root/triton"
  export TORCH_HOME="$cache_root/torch"
  export CUDA_CACHE_PATH="$cache_root/cuda"
  export XDG_CACHE_HOME="$cache_root/xdg"
  export TMPDIR="$tmp_root"
}

_aiinffra_activate_gpu_sandbox
