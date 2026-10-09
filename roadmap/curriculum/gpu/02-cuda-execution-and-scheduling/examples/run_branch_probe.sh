#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${SCRIPT_DIR}/execution_and_scheduling.cu"
RESULT_DIR="$(mktemp -d "/tmp/branch-probe-XXXXXX")"
ARCH="${PROBE_ARCH:-sm_86}"
NVCC_BIN="${NVCC_BIN:-nvcc}"

if ! command -v "${NVCC_BIN}" >/dev/null 2>&1; then
    echo "nvcc not found; set NVCC_BIN or put nvcc on PATH" >&2
    exit 1
fi

EXE="${RESULT_DIR}/cuda-execution"
"${NVCC_BIN}" -O3 -std=c++17 -lineinfo --resource-usage \
    "-arch=${ARCH}" "${SOURCE}" -o "${EXE}" 2>&1 | tee "${RESULT_DIR}/compile.log"

"${EXE}" --branch-probe --self-test | tee "${RESULT_DIR}/self-test.log"
"${EXE}" --branch-probe 257 8 3 3 | tee "${RESULT_DIR}/small.log"

if command -v compute-sanitizer >/dev/null 2>&1; then
    compute-sanitizer --tool memcheck --error-exitcode=1 \
        "${EXE}" --branch-probe 257 8 3 3 2>&1 | tee "${RESULT_DIR}/memcheck.log"
else
    echo "compute-sanitizer: SKIP (not found)" | tee "${RESULT_DIR}/memcheck.log"
fi

CUOBJDUMP_BIN="${CUOBJDUMP_BIN:-cuobjdump}"
if ! command -v "${CUOBJDUMP_BIN}" >/dev/null 2>&1; then
    nvcc_path="$(command -v "${NVCC_BIN}")"
    nvcc_dir="$(cd -- "$(dirname -- "${nvcc_path}")" && pwd)"
    if [[ -x "${nvcc_dir}/cuobjdump" ]]; then
        CUOBJDUMP_BIN="${nvcc_dir}/cuobjdump"
    fi
fi
if command -v "${CUOBJDUMP_BIN}" >/dev/null 2>&1 || [[ -x "${CUOBJDUMP_BIN}" ]]; then
    "${CUOBJDUMP_BIN}" --dump-sass "${EXE}" > "${RESULT_DIR}/branch_probe.sass.txt"
else
    echo "cuobjdump: SKIP (not found)" | tee "${RESULT_DIR}/sass.log"
fi

for steps in 1 32 256; do
    "${EXE}" --branch-probe 1048576 "${steps}" 50 9 \
        | tee "${RESULT_DIR}/large-steps-${steps}.log"
done

echo "branch_probe results: ${RESULT_DIR}"
