#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: profile-transpose-nsys.sh --personal-root ABS_PATH [--repo-root ABS_PATH]
       [--python PATH] [--rows N] [--cols N] [--warmup N] [--repeats N]
       [--impl triton|torch|both]
Run this script inside a tmux session using its personal socket.
EOF
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
abs_path() { [[ $1 = /* ]]; }
positive() { [[ $1 =~ ^[1-9][0-9]*$ ]]; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 2
default_repo=$(cd -- "$script_dir/.." && pwd -P) || exit 2
repo_root=$default_repo
personal_root=
python_bin=$(command -v python || true)
rows=4096 cols=4096 warmup=20 repeats=100 impl=both
while (($#)); do
  case "$1" in
    --personal-root|--repo-root|--python|--rows|--cols|--warmup|--repeats|--impl)
      (($# >= 2)) || die "missing value for $1"
      key=$1 value=$2; shift 2
      case "$key" in
        --personal-root) personal_root=$value ;;
        --repo-root) repo_root=$value ;;
        --python) python_bin=$value ;;
        --rows) rows=$value ;;
        --cols) cols=$value ;;
        --warmup) warmup=$value ;;
        --repeats) repeats=$value ;;
        --impl) impl=$value ;;
      esac ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n $personal_root ]] || die "--personal-root is required"
abs_path "$personal_root" || die "--personal-root must be absolute"
abs_path "$repo_root" || die "--repo-root must be absolute"
[[ -n $python_bin ]] || die "python was not found; pass --python explicitly"
for value in "$rows" "$cols" "$warmup" "$repeats"; do positive "$value" || die "numeric options must be positive integers"; done
[[ $impl == triton || $impl == torch || $impl == both ]] || die "invalid --impl"

[[ -d $personal_root && ! -L $personal_root ]] || die "personal root must be an existing non-symlink directory"
personal_root=$(cd -- "$personal_root" && pwd -P) || exit 2
[[ $personal_root != / && $personal_root != /root && $personal_root != /home && $personal_root != /tmp ]] || die "personal root is too broad"
[[ -d $repo_root ]] || die "repository root does not exist"
repo_root=$(cd -- "$repo_root" && pwd -P) || exit 2
[[ $repo_root == "$personal_root"/* ]] || die "repository root must be inside personal root"
[[ -f $repo_root/solutions/triton/transpose_profile.py && -f $repo_root/solutions/triton/transpose_v2_leetgpu.py ]] || die "transpose source files are missing from repo root"
abs_path "$python_bin" || die "--python must be an absolute executable path"
[[ -x $python_bin ]] || command -v "$python_bin" >/dev/null 2>&1 || die "python executable was not found"
command -v tmux >/dev/null 2>&1 || die "tmux is unavailable"
[[ -n ${TMUX:-} ]] || die "run inside the personal tmux session"

socket_path=${TMUX%%,*}
expected_socket=$personal_root/.tmux.sock
[[ $socket_path == "$expected_socket" ]] || die "current tmux socket does not match $expected_socket"
[[ -S $expected_socket ]] || die "personal tmux socket is missing"
socket_owner=$(stat -c %u "$expected_socket" 2>/dev/null || stat -f %u "$expected_socket" 2>/dev/null) || die "cannot verify personal socket owner"
[[ $socket_owner == "$(id -u)" ]] || die "personal socket is not owned by current user"
tmux -f /dev/null -S "$expected_socket" display-message -p '#{socket_path}' >/dev/null 2>&1 || die "personal tmux socket is not responsive"

umask 077
runs_dir=$personal_root/runs
if [[ -e $runs_dir || -L $runs_dir ]]; then
  [[ -d $runs_dir && ! -L $runs_dir ]] || die "runs must be a real directory directly under personal root"
  runs_dir=$(cd -- "$runs_dir" && pwd -P) || die "cannot resolve runs directory"
  [[ $runs_dir == "$personal_root/runs" ]] || die "runs directory resolves outside personal root"
else
  mkdir -- "$runs_dir" || die "cannot create runs directory"
  chmod 700 "$runs_dir"
fi

task_id="transpose-nsys-$(date -u +%Y%m%dT%H%M%SZ)-$$"
task_dir=$runs_dir/$task_id
mkdir -- "$task_dir" || die "task directory already exists: $task_dir"
chmod 700 "$task_dir"
[[ ! -L $task_dir && $(cd -- "$task_dir" && pwd -P) == "$task_dir" ]] || die "task directory escaped personal root"
profile_rc=125 stats_rc=125 tee_rc=0
write_exit_receipt() {
  local result=$?
  trap - EXIT
  printf 'profile_exit=%s\nstats_exit=%s\ntee_exit=%s\nexit=%s\n' \
    "$profile_rc" "$stats_rc" "$tee_rc" "$result" > "$task_dir/exitcode.txt" 2>/dev/null || true
  exit "$result"
}
trap write_exit_receipt EXIT

for subdir in tmp xdg xdg-cache cuda-cache torch triton pip-cache pycache; do
  target=$task_dir/$subdir
  mkdir -- "$target"
  chmod 700 "$target"
  [[ ! -L $target && $(cd -- "$target" && pwd -P) == "$target" ]] || die "task subdirectory escaped task root: $subdir"
done
export NSYS_TMPDIR="$task_dir/tmp" TMPDIR="$task_dir/tmp" TMP="$task_dir/tmp" TEMP="$task_dir/tmp"
export XDG_CONFIG_HOME="$task_dir/xdg" XDG_CACHE_HOME="$task_dir/xdg-cache"
export CUDA_CACHE_PATH="$task_dir/cuda-cache" TORCH_HOME="$task_dir/torch"
export TRITON_CACHE_DIR="$task_dir/triton" PIP_CACHE_DIR="$task_dir/pip-cache"
export PYTHONPYCACHEPREFIX="$task_dir/pycache" PYTHONDONTWRITEBYTECODE=1 HISTFILE=/dev/null
printf '%s\n' "$task_id" > "$task_dir/taskid.txt"
{
  printf 'task_id=%s\npersonal_root=%s\nrepo_root=%s\n' "$task_id" "$personal_root" "$repo_root"
  git -C "$repo_root" rev-parse HEAD 2>&1 | sed 's/^/git_head=/'
  git -C "$repo_root" status --short 2>&1 | sed 's/^/git_dirty=/'
  sha256sum "$repo_root/solutions/triton/transpose_v2_leetgpu.py" 2>&1 | sed 's/^/kernel_sha256=/'
  "$python_bin" --version 2>&1 | sed 's/^/python_version=/'
  nsys --version 2>&1 | sed 's/^/nsys_version=/'
} > "$task_dir/environment.txt"

profile_args=(--rows "$rows" --cols "$cols" --warmup "$warmup" --repeats "$repeats" --impl "$impl" --capture-profiler-api)
cmd=(nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output "$task_dir/transpose" "$python_bin" -B "$repo_root/solutions/triton/transpose_profile.py" "${profile_args[@]}")
{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
  printf 'cd %q\n' "$repo_root"
  for name in NSYS_TMPDIR TMPDIR TMP TEMP XDG_CONFIG_HOME XDG_CACHE_HOME CUDA_CACHE_PATH TORCH_HOME TRITON_CACHE_DIR PIP_CACHE_DIR PYTHONPYCACHEPREFIX PYTHONDONTWRITEBYTECODE HISTFILE; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
  printf 'exec '
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$task_dir/command.sh"
chmod 700 "$task_dir/command.sh"

set +e
"${cmd[@]}" 2>&1 | tee "$task_dir/task.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
profile_rc=${pipeline_status[0]:-125}
tee_rc=${pipeline_status[1]:-125}
printf '%s\n' "$profile_rc" > "$task_dir/profile.exitcode"
printf '%s\n' "$tee_rc" > "$task_dir/tee.exitcode"

stats_rc=125
report=$task_dir/transpose.nsys-rep
if ((profile_rc == 0 && tee_rc == 0)) && grep -Fq 'PROFILE_TARGET_COMPLETE correctness=PASS' "$task_dir/task.log" && [[ -s $report && ! -L $report ]]; then
  if nsys stats --report cuda_gpu_kern_sum --report nvtx_sum --format column "$report" > "$task_dir/stats.txt" 2> "$task_dir/stats.stderr"; then
    stats_rc=0
    case "$impl" in
      triton) required_nvtx=(transpose/triton) ;;
      torch) required_nvtx=(transpose/pytorch) ;;
      both) required_nvtx=(transpose/triton transpose/pytorch) ;;
    esac
    for label in "${required_nvtx[@]}"; do
      if ! grep -Fq -- "$label" "$task_dir/stats.txt"; then
        printf 'required NVTX label missing from stats: %s\n' "$label" >> "$task_dir/stats.stderr"
        stats_rc=1
      fi
    done
    if [[ $impl == triton || $impl == both ]] && ! grep -Fq -- matrix_transpose_kernel "$task_dir/stats.txt"; then
      printf 'Triton kernel missing from CUDA summary: matrix_transpose_kernel\n' >> "$task_dir/stats.stderr"
      stats_rc=1
    fi
  else
    stats_rc=$?
  fi
else
  printf 'stats skipped: profile, tee, completion marker, or report check failed\n' > "$task_dir/stats.stderr"
fi
printf 'stats_exit=%s\n' "$stats_rc" > "$task_dir/stats.exitcode"
final_rc=0
((profile_rc == 0 && stats_rc == 0 && tee_rc == 0)) || final_rc=1
printf 'TASK_DIR=%s\nPROFILE_EXIT=%s\nSTATS_EXIT=%s\nEXIT=%s\n' "$task_dir" "$profile_rc" "$stats_rc" "$final_rc"
exit "$final_rc"
