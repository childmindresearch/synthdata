"""Measure how GPU training throughput changes with the number of fits sharing a GPU.

Starts k processes at once, each fitting the same synthcity model for a fixed
number of epochs on a CSV, for each k given, and prints the wall time and the
total epochs per minute. Throughput that keeps rising with k means the GPU
was not full; flat throughput means extra fits only take turns. Use it to pick
``compute.fits_per_gpu`` for a dataset and GPU. Run it on an idle GPU.

    uv run python scripts/gpu_concurrency.py --csv <train_imputed.csv> --model ctgan
"""

import argparse
import multiprocessing as mp
import os
import time


def _fit(csv, model, epochs, device, barrier, out):
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import pandas as pd
    import torch
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader

    torch.set_num_threads(4)
    data = GenericDataLoader(pd.read_csv(csv))
    plugin = Plugins().get(model, n_iter=epochs, device=torch.device(device))
    barrier.wait()
    started = time.monotonic()
    plugin.fit(data)
    out.put(time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", default="ctgan")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fits", default="1,2,3,4,6")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    ctx = mp.get_context("spawn")
    print(f"model={args.model} epochs={args.epochs} device={args.device}")
    print("fits_at_once,wall_s,mean_fit_s,epochs_per_min,speedup_vs_1")
    base = None
    for k in (int(n) for n in args.fits.split(",")):
        barrier, out = ctx.Barrier(k + 1), ctx.Queue()
        procs = [
            ctx.Process(
                target=_fit, args=(args.csv, args.model, args.epochs, args.device, barrier, out)
            )
            for _ in range(k)
        ]
        for p in procs:
            p.start()
        barrier.wait()
        started = time.monotonic()
        times = [out.get() for _ in range(k)]
        wall = time.monotonic() - started
        for p in procs:
            p.join()
        rate = k * args.epochs / wall * 60
        base = base or rate
        print(f"{k},{wall:.0f},{sum(times) / k:.0f},{rate:.1f},{rate / base:.2f}", flush=True)


if __name__ == "__main__":
    main()
