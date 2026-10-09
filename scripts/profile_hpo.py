#!/usr/bin/env python
"""CLI: time a few HPO trials per synthcity model to set ``hpo`` budgets.

Runs the real HPO objective (fit on search-train, generate ``n_samples`` rows,
score on tuning) for ``--trials`` trials per model with training length pinned
(``PROFILE_EPOCHS``), no pruning and no timeout, then writes timings only:

    <out>/system.json   machine, library versions, split sizes (counts only)
    <out>/trials.csv    one row per trial: model, state, seconds, params, score
    <out>/summary.csv   per model: mean trial time, time per epoch/tree, peak GPU
                        memory, and the cost of the default (untuned) fit
    <out>/gpu.csv       nvidia-smi utilisation samples (when nvidia-smi exists)

No data rows are written. Imputation must have run: pass ``--impute`` or run
``synthdata-impute --config <path>`` first.

Usage:
    python -m scripts.profile_hpo --config configs/config_sim.yaml [--impute] [--trials 2] [--models ctgan arf]
"""

import argparse
import inspect
import json
import os
import platform
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

from synthdata.config import load_config
from synthdata.data import load_dataset, load_imputed_splits
from synthdata.generation import hpo as hpo_mod
from synthdata.generation import synthcity_backend as sc
from synthdata.imputation import run_imputation
from synthdata.utils import ensure_dir, get_logger, resolve_device, set_global_seed

logger = get_logger("profile_hpo")

#: Pinned training length per trial (PATE-GAN: iterations of 10 GAN epochs);
#: other models that take ``n_iter`` use 50.
#: ARF's tree count stays searched, so its trials span the 10-100 range.
PROFILE_EPOCHS = {"pategan": 6, "ddpm": 100}
DEFAULT_EPOCHS = 50
#: Trial params that measure training length, for the per-unit time.
LENGTH_PARAMS = ("n_iter", "num_trees", "n_estimators")
PACKAGES = ("synthdata", "synthcity", "torch", "optuna", "tabpfn", "arfpy", "xgboost")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def system_info(dataset, device: str) -> dict:
    import torch

    gpus = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        ram_gb = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(ram_gb, 1) if ram_gb else None,
        "device": device,
        "gpus": gpus,
        "cuda": torch.version.cuda,
        "packages": {p: _version(p) for p in PACKAGES},
        "search_train_shape": list(dataset.search_train_imputed_df.shape),
        "tuning_shape": list(dataset.tuning_imputed_df.shape),
        "train_shape": list(dataset.train_imputed_df.shape),
    }


def default_length(name: str, hpo_cfg) -> int | None:
    """What the untuned (default) model trains for: the plugin's own length,
    capped at the top of the searched range (``sc.untuned_params``)."""
    capped = sc.untuned_params(name, hpo_cfg).get("n_iter")
    if capped is not None:
        return capped
    params = inspect.signature(sc.get_plugin_class(name).__init__).parameters
    return next((params[k].default for k in LENGTH_PARAMS if k in params), None)


def trial_rows(name: str, study) -> list[dict]:
    rows = []
    for t in study.trials:
        seconds = (
            (t.datetime_complete - t.datetime_start).total_seconds()
            if t.datetime_complete and t.datetime_start
            else None
        )
        length = next((t.params[k] for k in LENGTH_PARAMS if k in t.params), None)
        rows.append(
            {
                "model": name,
                "trial": t.number,
                "state": t.state.name,
                "seconds": seconds,
                "length": length,
                "batch_size": t.params.get("batch_size"),
                "value": t.value,
                "params": json.dumps(t.params, default=str),
                "error": t.user_attrs.get("error"),
            }
        )
    return rows


def _recording_errors(objective):
    """Store a failed trial's exception (type and message) so trials.csv shows why."""

    def wrapped(trial):
        try:
            return objective(trial)
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}"[:500])
            logger.exception("[%s] trial %d failed", trial.study.study_name, trial.number)
            raise

    return wrapped


def summarize(trials: pd.DataFrame, peak_gb: dict, default_lengths: dict) -> pd.DataFrame:
    rows = []
    for name, group in trials.groupby("model", sort=False):
        done = group[group["state"] == "COMPLETE"]
        per_unit = (
            (done["seconds"] / done["length"]).mean() if done["length"].notna().any() else None
        )
        length = default_lengths.get(name)
        rows.append(
            {
                "model": name,
                "trials_complete": len(done),
                "trials_failed": int((group["state"] == "FAIL").sum()),
                "mean_trial_s": done["seconds"].mean(),
                "max_trial_s": done["seconds"].max(),
                "s_per_length_unit": per_unit,
                "peak_gpu_gb": peak_gb.get(name),
                "default_length": length,
                "default_fit_h_est": per_unit * length / 3600 if per_unit and length else None,
            }
        )
    return pd.DataFrame(rows)


def start_gpu_sampler(path: Path) -> subprocess.Popen | None:
    if shutil.which("nvidia-smi") is None:
        return None
    query = "timestamp,index,name,utilization.gpu,memory.used,memory.total"
    return subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv", "-l", "15", "-f", str(path)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--trials", type=int, default=2, help="Timed trials per model.")
    parser.add_argument(
        "--models", nargs="*", default=None, help="synthcity models (default: the config's)."
    )
    parser.add_argument("--impute", action="store_true", help="Run (and time) imputation first.")
    parser.add_argument(
        "--out", default=None, help="Output folder (default: output/<name>/profile)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = ensure_dir(args.out or Path("output") / cfg.name / "profile")
    set_global_seed(cfg.seed)
    hpo_cfg = cfg.generation.hpo
    hpo_cfg.enabled = True
    hpo_cfg.pruner = None  # every trial runs to its pinned length
    hpo_cfg.storage = f"sqlite:///{out / 'optuna_profile.db'}"
    models = args.models or cfg.generation.synthcity.names
    # Read before the profiling trials' training length is pinned below.
    default_lengths = {name: default_length(name, hpo_cfg) for name in models}
    for name in models:
        hpo_cfg.n_trials_per_model[name] = args.trials
        hpo_cfg.timeout_seconds_per_model[name] = None
        if sc.plugin_accepts(name, "n_iter"):
            epochs = PROFILE_EPOCHS.get(name, DEFAULT_EPOCHS)
            hpo_cfg.epoch_ranges[name] = [epochs, epochs, 1]

    impute_seconds = None
    if args.impute:
        started = time.time()
        run_imputation(cfg, load_dataset(cfg))  # reuses its cache when imputation.cache is on
        impute_seconds = round(time.time() - started, 1)

    started = time.time()
    dataset = load_imputed_splits(load_dataset(cfg))
    if dataset.search_train_imputed_df is None or dataset.tuning_imputed_df is None:
        raise SystemExit(
            "No imputed search/tuning data. Run `synthdata-impute --config <path>` first."
        )
    device = resolve_device(cfg.device)
    info = system_info(dataset, device)
    info["load_seconds"] = round(time.time() - started, 1)
    info["impute_seconds"] = impute_seconds
    info["imputation_method"] = cfg.imputation.method
    (out / "system.json").write_text(json.dumps(info, indent=2))
    logger.info("system: %s", info)

    workspace = out / "synthcity_workspace"
    classification = dataset.target_is_categorical
    match_prior = cfg.generation.match_class_prior and classification
    eval_fn = hpo_mod.build_hpo_eval_fn(
        dataset.search_train_imputed_df,
        dataset.tuning_imputed_df,
        dataset.target_column,
        dataset.nominal_columns,
        dataset.categorical_columns,
        classification,
        dataset.sensitive_columns,
        hpo_cfg,
        cfg.seed,
        match_prior=match_prior,
        workspace=workspace,
    )
    fairness_column = dataset.protected_columns[0] if dataset.protected_columns else None
    loader = sc.make_loader(
        dataset.search_train_imputed_df,
        dataset.target_column,
        dataset.sensitive_columns,
        random_state=cfg.seed,
        fairness_column=fairness_column,
    )

    import torch

    sampler = start_gpu_sampler(out / "gpu.csv")
    rows, peak_gb = [], {}
    try:
        for name in models:
            logger.info("[%s] profiling %d trial(s)", name, args.trials)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            objective = sc.build_synthcity_objective(
                name,
                loader,
                hpo_cfg,
                cfg.seed,
                eval_fn,
                cfg.generation.n_samples,
                workspace=workspace,
                device=device,
                classification=classification,
                class_prior=(
                    dataset.search_train_imputed_df[dataset.target_column].value_counts(
                        normalize=True
                    )
                    if match_prior
                    else None
                ),
                discrete_columns=dataset.all_categorical_columns,
            )
            try:
                hpo_mod.run_study(
                    f"hpo_{name}", _recording_errors(objective), hpo_cfg, out, cfg.seed
                )
            except Exception:  # keep profiling the other models
                logger.exception("[%s] study failed", name)
            if torch.cuda.is_available():
                peak_gb[name] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            study = hpo_mod.create_study(f"hpo_{name}", hpo_cfg, out, cfg.seed)
            rows.extend(trial_rows(name, study))
            trials = pd.DataFrame(rows)
            trials.to_csv(out / "trials.csv", index=False)
            summarize(trials, peak_gb, default_lengths).to_csv(out / "summary.csv", index=False)
    finally:
        if sampler is not None:
            sampler.terminate()
    logger.info("Done in %.1f h. Send back: %s", (time.time() - started) / 3600, out)


if __name__ == "__main__":
    main()
