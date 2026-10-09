# Branch probe validation — 2026-10-09

- `bash -n examples/run_branch_probe.sh`: exit 0.
- Windows `nvcc -O3 -std=c++17 -lineinfo --resource-usage -arch=sm_86 ... -o %TEMP%/.../branch_probe.exe`: exit 0.
- `branch_probe.exe --branch-probe --self-test`: exit 0, `branch_probe self-test PASS`.
- Self-test covers flag counts for n=32/64/33/257, finite CPU references, invalid parameters, guard corruption and NaN detection.
- Final local executable: `C:\Users\yq\AppData\Local\Temp\aiinffra-branch-probe-20261009\branch_probe.exe`.
- Compile command: `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe" -O3 -std=c++17 -lineinfo --resource-usage -arch=sm_86 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/execution_and_scheduling.cu -o "C:\Users\yq\AppData\Local\Temp\aiinffra-branch-probe-20261009\branch_probe.exe"` (exit 0).
- CPU command: `"C:\Users\yq\AppData\Local\Temp\aiinffra-branch-probe-20261009\branch_probe.exe" --branch-probe --self-test` (exit 0, `branch_probe self-test PASS`).
- SASS command: `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\cuobjdump.exe" --dump-sass "C:\Users\yq\AppData\Local\Temp\aiinffra-branch-probe-20261009\branch_probe.exe" > "C:\Users\yq\AppData\Local\Temp\aiinffra-branch-probe-20261009\branch_probe.sass.txt"` (exit 0).
- SASS inspection found conditional `BRA` selection and two `CALL.REL.NOINC` sites in `branch_probe_kernel`; the positive-chain path contains `FFMA` with `+0.00001` and a loop `BRA`, while the negative-chain path contains `FFMA` with `-0.00001` and a separate loop `BRA`. These are static instruction facts only. `nvidia-smi -L` was unavailable, so no GPU probe, memcheck or performance result is claimed; run `bash examples/run_branch_probe.sh` after pulling on the RTX 3090 server.
