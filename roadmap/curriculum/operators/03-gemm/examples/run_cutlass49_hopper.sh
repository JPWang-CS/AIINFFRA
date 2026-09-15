#!/usr/bin/env bash
set -euo pipefail

# Build and run NVIDIA CUTLASS v3.8.0 example 49 for Hopper SM90a.
# Every invocation re-runs CMake so the build cache is explicitly set to 90a.

if [[ -z "${CUTLASS_ROOT:-}" ]]; then
  echo "Set CUTLASS_ROOT to a clean NVIDIA/cutlass v3.8.0 checkout." >&2
  exit 2
fi

tag="$(git -C "$CUTLASS_ROOT" describe --tags --exact-match HEAD 2>/dev/null || true)"
if [[ "$tag" != "v3.8.0" ]]; then
  echo "Expected exact CUTLASS tag v3.8.0; found '${tag:-no exact tag}'." >&2
  exit 2
fi

commit="$(git -C "$CUTLASS_ROOT" rev-parse HEAD)"
if [[ "$commit" != "afa1772203677c5118fcd82537a9c8fefbcc7008" ]]; then
  echo "Expected CUTLASS v3.8.0 commit afa1772203677c5118fcd82537a9c8fefbcc7008; found '$commit'." >&2
  exit 2
fi

if [[ -n "$(git -C "$CUTLASS_ROOT" status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing a dirty or untracked checkout at the pinned v3.8.0 commit; use an unmodified release tree." >&2
  exit 2
fi

build_dir="${CUTLASS_BUILD_DIR:-${CUTLASS_ROOT}-sm90a-build}"

cmake -S "$CUTLASS_ROOT" -B "$build_dir" \
  -DCUTLASS_NVCC_ARCHS=90a \
  -DCUTLASS_ENABLE_TESTS=OFF

cmake --build "$build_dir" \
  --target 49_collective_builder \
  --parallel "${JOBS:-4}"

binary=""
for candidate in \
  "$CUTLASS_ROOT/examples/49_hopper_gemm_with_collective_builder/49_collective_builder" \
  "$build_dir/examples/49_hopper_gemm_with_collective_builder/49_collective_builder" \
  "$build_dir/49_collective_builder"; do
  if [[ -x "$candidate" ]]; then
    binary="$candidate"
    break
  fi
done

if [[ -z "$binary" ]]; then
  echo "Build succeeded but 49_collective_builder was not found in expected locations." >&2
  exit 3
fi

if [[ "$#" -eq 0 ]]; then
  set -- --m=2048 --n=2048 --k=2048 --l=1 --alpha=1 --beta=0
fi

exec "$binary" "$@"
