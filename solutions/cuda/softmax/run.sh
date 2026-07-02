#!/usr/bin/env bash
# ============================================================
# softmax solve() 本地测试脚本（服务器跑）
#
# 用法：
#   ./run.sh                       # 编译 softmax_naive.cu + main.cu，跑默认几个 N
#   ./run.sh 500000                # 只跑指定 N（LeetGPU 题面 max）
#   KERNEL=softmax_online.cu ./run.sh     # 测后续其它版本（同签名 solve）
#   ARCH=-arch=sm_89 ./run.sh             # 指定 GPU 架构（默认 -arch=native 自动探测）
#   NVCC=/path/to/nvcc ./run.sh           # 指定 nvcc 路径
#
# 退出码：0 = 编译+运行成功（不代表精度 PASS，精度看输出 [PASS]/[FAIL]）
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

NVCC=${NVCC:-nvcc}
KERNEL=${KERNEL:-softmax_naive.cu}
ARCH=${ARCH:--arch=native}
OUT=${OUT:-test_softmax}

if [ ! -f "$KERNEL" ]; then
    echo "[error] kernel file '$KERNEL' not found in $(pwd)" >&2
    echo "        expect e.g. softmax_naive.cu / softmax_online.cu" >&2
    exit 1
fi

echo "[build] $NVCC $ARCH -O2 -std=c++14 -DKERNEL_FILE=\"$KERNEL\" main.cu $KERNEL -o $OUT"
"$NVCC" $ARCH -O2 -std=c++14 "-DKERNEL_FILE=\"$KERNEL\"" main.cu "$KERNEL" -o "$OUT"

echo "[run] ./$OUT $*"
./"$OUT" "$@"
