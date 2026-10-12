"""Measure how GPU training throughput changes with the number of fits sharing a GPU.

Starts k processes at once, each fitting the same synthcity model for a fixed
number of epochs on a CSV, for each k given, and prints the wall time and the
total epochs per minute. Throughput that keeps rising with k means the GPU
was not full; flat throughput means extra fits only take turns. Use it to pick
``compute.fits_per_gpu`` for a dataset and GPU. Run it on an idle GPU.

    uv run python scripts/gpu_concurrency.py --csv <train_imputed.csv> --model ctgan

With ``--background``, one fit is ``--model`` and the other k - 1 are the
background model, to see how much sharing slows a model down (its time
against k = 1) and how fast the others train meanwhile.

    uv run python scripts/gpu_concurrency.py --csv <csv> --model ddpm --epochs 100 \
        --background ctgan --background-epochs 40
"""

import argparse
import multiprocessing as mp
import os
import time


def _fit(csv, model, epochs, device, barrier, out, role="main"):
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
    out.put((role, time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", default="ctgan")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fits", default="1,2,3,4,6")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--background", help="model for the other k - 1 fits")
    parser.add_argument("--background-epochs", type=int, default=40)
    args = parser.parse_args()
    ctx = mp.get_context("spawn")
    print(f"model={args.model} epochs={args.epochs} device={args.device}")
    if args.background:
        print(f"background={args.background} epochs={args.background_epochs}")
        print("fits_at_once,main_fit_s,main_slowdown_vs_1,background_mean_fit_s")
    else:
        print("fits_at_once,wall_s,mean_fit_s,epochs_per_min,speedup_vs_1")
    base = None
    for k in (int(n) for n in args.fits.split(",")):
        barrier, out = ctx.Barrier(k + 1), ctx.Queue()
        jobs = [(args.model, args.epochs, "main")]
        other = args.background or args.model
        other_epochs = args.background_epochs if args.background else args.epochs
        jobs += [(other, other_epochs, "background" if args.background else "main")] * (k - 1)
        procs = [
            ctx.Process(target=_fit, args=(args.csv, m, e, args.device, barrier, out, role))
            for m, e, role in jobs
        ]
        for p in procs:
            p.start()
        barrier.wait()
        started = time.monotonic()
        results = [out.get() for _ in range(k)]
        wall = time.monotonic() - started
        for p in procs:
            p.join()
        mains = [t for role, t in results if role == "main"]
        if args.background:
            others = [t for role, t in results if role == "background"]
            base = base or mains[0]
            bg = f"{sum(others) / len(others):.0f}" if others else ""
            print(f"{k},{mains[0]:.0f},{mains[0] / base:.2f},{bg}", flush=True)
        else:
            rate = k * args.epochs / wall * 60
            base = base or rate
            print(f"{k},{wall:.0f},{sum(mains) / k:.0f},{rate:.1f},{rate / base:.2f}", flush=True)


if __name__ == "__main__":
    main()
