"""Explicitly launched multi-GPU NCCL correctness and event-window probe."""
import argparse
from datetime import timedelta
import os


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--elements",type=int,default=262144)
    args=parser.parse_args()
    if args.elements<=0:
        raise ValueError("positive element count required")
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print("SKIP: PyTorch unavailable; no collective ran")
        return
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable; no collective ran")
        return
    if "RANK" not in os.environ:
        raise RuntimeError("launch this script with torchrun")
    local_rank=int(os.environ["LOCAL_RANK"])
    local_world=int(os.environ.get("LOCAL_WORLD_SIZE","1"))
    if local_world>torch.cuda.device_count():
        raise RuntimeError("not enough visible GPUs for local ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl",timeout=timedelta(seconds=60))
    try:
        rank,world=dist.get_rank(),dist.get_world_size()
        if world<2:
            raise RuntimeError("use at least two ranks for this multi-GPU probe")
        data=torch.full((args.elements,),rank+1.0,device="cuda",dtype=torch.float32)
        dist.all_reduce(data)
        torch.testing.assert_close(data,torch.full_like(data,world*(world+1)/2),rtol=0,atol=0)
        data.zero_()
        for _ in range(5):dist.all_reduce(data)
        dist.barrier()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):dist.all_reduce(data)
        end.record();end.synchronize()
        elapsed=torch.tensor([start.elapsed_time(end)/20],device="cuda")
        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        if rank==0:
            ms=elapsed.item()
            algorithm_gbs=data.numel()*data.element_size()/(ms*1e6)
            print(f"PASS: AllReduce SUM, ranks={world}, elements={args.elements}, dtype=FP32")
            print(f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} NCCL={torch.cuda.nccl.version()}")
            print(f"max rank event window={ms:.6f} ms, algorithm GB/s={algorithm_gbs:.3f}")
            print("Event window includes issue gaps; ring-equivalent bus factor is not a measured link counter.")
    finally:
        dist.destroy_process_group()


if __name__=="__main__":
    main()
