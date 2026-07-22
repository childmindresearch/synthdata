"""RefiDiff imputation backend: warm-up refinement + Mamba/EDM diffusion + polishing.

Ported (not vendored) from Atik-Ahamed/RefiDiff (Apache-2.0), the reference
implementation of:

    Ahamed, A. et al. "RefiDiff: Refinement-Aware Diffusion for Efficient
    Missing Data Imputation." arXiv:2505.14451, 2025.
    https://arxiv.org/abs/2505.14451
    https://github.com/Atik-Ahamed/RefiDiff

Only the core algorithm is ported (``model.py``'s ``CustomDenoiser``/
``MambaBlock``/``Precond``/``Model``, ``diffusion_utils.py``'s ``EDMLoss``/
``impute_mask``/``refinement``, and ``dataset.py``'s ``mean_std``/binary
categorical encoding). The upstream repo's file-based dataset loading,
benchmark evaluation harness, and baseline-comparison scripts are not used
here; this module operates on in-memory DataFrames via ``impute_dataframe``,
matching the contract of :mod:`synthdata.imputation.tabimpute_backend`.

Code below is adapted/modified from the upstream sources (variable/class
names, DataFrame plumbing, checkpointing, logging, and the MLP fallback
denoiser are new; the diffusion math, warm-up refinement logic, and
Mamba-based denoiser architecture follow the paper's reference code closely),
per the Apache-2.0 license's attribution requirement.

Deliberate deviations from the upstream reference implementation:

- Binary category decoding clips out-of-range indices (a bit pattern can
  decode to a value >= the number of observed categories) to the nearest
  valid index, logging a warning -- the upstream code has no such guard.
- Training checkpoints every ``RefiDiffConfig.checkpoint_every`` epochs
  (optimizer + scheduler + epoch/best_loss/patience state) so an interrupted
  run (e.g. shared-GPU preemption) resumes instead of retraining from
  scratch -- the upstream reference script has no checkpointing.
- Both numeric *and* categorical columns are fully de-standardized back to
  raw units before being written into the output DataFrame. The upstream
  benchmark script's own MAE/RMSE evaluation reports numeric columns in
  z-scored (not raw) units, which is fine for its own relative benchmark
  comparisons but wrong for a general-purpose imputation API that must
  return values in the original feature's units.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from synthdata.config import RefiDiffConfig
from synthdata.data import decode_label_encoded_columns, label_encode_non_numeric_columns
from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# EDM/VE-SDE sampling schedule constants (Karras et al. 2022 defaults, as
# used by RefiDiff's diffusion_utils.py).
_SIGMA_MIN = 0.002
_SIGMA_MAX = 80.0
_RHO = 7
_S_CHURN = 1
_S_MIN = 0
_S_MAX = float("inf")
_S_NOISE = 1
_N_LANGEVIN = 10  # inner correction steps per outer diffusion step (impute_mask's N)

# CatBoostClassifier's own default (1000) is ~10x the cost this warm-up fill
# needs; see _warmup_refine's docstring for the empirical timing behind this.
_CATBOOST_WARMUP_ITERATIONS = 100


# ---------------------------------------------------------------------------
# Binary categorical encoding (memory-efficient alternative to one-hot: each
# categorical column uses ceil(log2(n_categories)) bit-columns instead of
# n_categories one-hot columns).
# ---------------------------------------------------------------------------


def _fit_categorical_binary_encoders(df: pd.DataFrame, categorical_columns: list) -> dict:
    """Build a category<->index<->binary-code map per categorical column."""
    encoders = {}
    for col in categorical_columns:
        categories = sorted(df[col].dropna().unique().tolist(), key=str)
        n_categories = max(len(categories), 1)
        n_bits = max((n_categories - 1).bit_length(), 1)
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        idx_to_cat = {idx: cat for cat, idx in cat_to_idx.items()}
        encoders[col] = {
            "cat_to_idx": cat_to_idx,
            "idx_to_cat": idx_to_cat,
            "n_bits": n_bits,
            "n_categories": n_categories,
        }
    return encoders


def _encode_categorical_to_bits(series: pd.Series, encoder: dict) -> tuple:
    """Encode one categorical column into a (n_rows, n_bits) 0/1 float matrix.

    Missing entries are encoded as all-zero bits (a placeholder; the
    corresponding positions are always masked as missing downstream, so this
    placeholder value is never treated as ground truth).
    """
    n_bits = encoder["n_bits"]
    cat_to_idx = encoder["cat_to_idx"]
    values = series.to_numpy()
    missing = series.isna().to_numpy()
    bits = np.zeros((len(series), n_bits), dtype=np.float64)
    for i, val in enumerate(values):
        if missing[i]:
            continue
        idx = cat_to_idx[val]
        bit_str = format(idx, f"0{n_bits}b")
        bits[i] = [int(b) for b in bit_str]
    return bits, missing


def _decode_bits_to_categorical(bits: np.ndarray, encoder: dict, column_name: str) -> np.ndarray:
    """Decode a (n_rows, n_bits) matrix back to original category values.

    A bit pattern is read as a single binary integer (matching the upstream
    encoding), so it can decode to an index >= n_categories (e.g. 2 bits
    encoding 3 categories can decode to index 3). Unlike the upstream repo,
    we clip such out-of-range indices to the nearest valid one and log how
    often this happened, rather than silently emitting an invalid category.
    """
    idx_to_cat = encoder["idx_to_cat"]
    n_categories = encoder["n_categories"]
    thresholded = (bits > 0.5).astype(int)
    idx = np.array(
        [int("".join(str(b) for b in row), 2) if row.size else 0 for row in thresholded],
        dtype=np.int64,
    )
    out_of_range = idx >= n_categories
    n_out_of_range = int(out_of_range.sum())
    if n_out_of_range:
        logger.warning(
            "refidiff: column %r decoded %d/%d out-of-range binary category indices "
            "(valid range [0, %d]); clipping to the nearest valid index",
            column_name,
            n_out_of_range,
            len(idx),
            n_categories - 1,
        )
        idx = np.clip(idx, 0, n_categories - 1)
    return np.array([idx_to_cat[i] for i in idx])


# ---------------------------------------------------------------------------
# Standardization and warm-up refinement (ported from dataset.py/diffusion_utils.py)
# ---------------------------------------------------------------------------


def _mean_std(data: np.ndarray, missing_mask: np.ndarray) -> tuple:
    """Per-column mean/std computed from observed (non-missing) entries only."""
    observed = (~missing_mask).astype(np.float64)
    denom = observed.sum(axis=0)
    denom[denom == 0] = 1
    mean = (data * observed).sum(axis=0) / denom
    var = ((data - mean) ** 2 * observed).sum(axis=0) / denom
    std = np.sqrt(var)
    n_constant = int((std == 0).sum())
    if n_constant:
        logger.debug(
            "refidiff: %d column(s) have zero observed-value variance; "
            "using std=1 for those columns to avoid divide-by-zero",
            n_constant,
        )
        std[std == 0] = 1
    return mean, std


def _warmup_refine(
    X: np.ndarray, missing_mask: np.ndarray, len_num: int, device: str = "cpu"
) -> np.ndarray:
    """Single-pass per-column imputation: XGBRegressor (numeric) / CatBoostClassifier (bit).

    Ported from RefiDiff's ``diffusion_utils.refinement()``. Each column's
    missing entries are predicted from all *other* columns' current values in
    one pass (not iterative MICE) -- other still-missing columns are read as
    whatever placeholder value they currently hold (0 on the first call,
    the diffusion model's output on the polishing-pass call).

    ``device`` controls GPU usage for the per-column model fits, and is
    deliberately asymmetric between the two libraries based on empirical
    profiling on data shaped like this function's actual workload (~3500
    rows, ~1000 input columns per fit): XGBRegressor's GPU path was ~41x
    faster than CPU (0.97s vs 40.4s per fit), while CatBoostClassifier's GPU
    path was consistently *slower* than CPU (~47-78s vs ~53s per fit across
    repeated fits and with an explicit ``Pool`` -- ruling out a one-time
    CUDA-context-init cost or CPU/GPU data-marshalling overhead as the
    cause; GPU histogram-building genuinely doesn't pay off for CatBoost at
    this row count). So: XGBRegressor uses GPU when available,
    CatBoostClassifier always runs on CPU regardless of ``device``.

    CatBoostClassifier is also capped at ``_CATBOOST_WARMUP_ITERATIONS``
    boosting rounds (its own default is 1000, ~10x more): on the same
    profiling shape, fit+predict time scaled ~linearly with iterations
    (1000 -> 28.9s, 200 -> 5.8s, 100 -> 2.9s, 50 -> 1.4s), dwarfing every
    other cost in this function once a dataset has many categorical
    columns. XGBRegressor's own sklearn default is already ~100 trees, so
    100 keeps roughly matched "boosting effort" between the two model
    types for what is only ever a quick warm-up/polishing fill -- the
    diffusion model (and, for the polishing pass, the fact that this refines
    the diffusion model's own output) is what drives final imputation
    quality, not this per-column fill.
    """
    from catboost import CatBoostClassifier
    from xgboost import XGBRegressor

    use_xgb_gpu = device == "cuda"
    X = X.copy()
    _, n_features = X.shape
    for col in tqdm(range(n_features), desc="refidiff warm-up refinement", unit="col"):
        missing_idx = np.where(missing_mask[:, col])[0]
        if len(missing_idx) == 0:
            continue
        observed_idx = np.where(~missing_mask[:, col])[0]
        if len(observed_idx) == 0:
            logger.warning(
                "refidiff: warm-up refinement column index %d has zero observed rows "
                "(entirely missing); leaving its placeholder fill value unchanged",
                col,
            )
            continue

        X_obs_input = np.delete(X[observed_idx], col, axis=1)
        y_obs = X[observed_idx, col]
        X_miss_input = np.delete(X[missing_idx], col, axis=1)

        if col >= len_num:
            unique_vals = sorted(set(y_obs.tolist()))
            if len(unique_vals) == 1:
                X[missing_idx, col] = unique_vals[0]
                continue
            val_to_label = {val: i for i, val in enumerate(unique_vals)}
            label_to_val = {i: val for val, i in val_to_label.items()}
            y_obs_mapped = np.array([val_to_label[v] for v in y_obs])
            model = CatBoostClassifier(
                logging_level="Silent",
                iterations=_CATBOOST_WARMUP_ITERATIONS,
                # Without this, CatBoost writes training-log files (learn_error.tsv,
                # time_left.tsv, etc.) under ./catboost_info/ on every fit -- this
                # runs once per categorical column (hundreds of times per impute
                # call), so it's both wasted I/O and an uncommitted stray artifact
                # left in the repo root.
                allow_writing_files=False,
            )
            model.fit(X_obs_input, y_obs_mapped, verbose=False)
            y_pred_labels = np.asarray(model.predict(X_miss_input)).reshape(-1)
            X[missing_idx, col] = np.array([label_to_val[int(v)] for v in y_pred_labels])
        else:
            model = XGBRegressor(tree_method="hist", device="cuda" if use_xgb_gpu else "cpu")
            model.fit(X_obs_input, y_obs)
            if use_xgb_gpu:
                # X_miss_input is a plain (CPU-resident) numpy array, so predicting
                # with the booster still configured for device="cuda" triggers an
                # internal "mismatched devices" DMatrix-fallback warning/overhead
                # (predict() would otherwise use the faster inplace_predict path
                # only when the booster's device matches the input's device).
                # The predict set here is small (one column's missing rows) so
                # this costs nothing measurable -- confirmed empirically (0.06s
                # with the fallback vs 0.018s without, next to a ~1s GPU fit).
                model.get_booster().set_param({"device": "cpu"})
            X[missing_idx, col] = model.predict(X_miss_input)
    return X


# ---------------------------------------------------------------------------
# Denoiser architectures
# ---------------------------------------------------------------------------


class PositionalEmbedding(nn.Module):
    """Sinusoidal noise-level embedding (ported from RefiDiff's model.py)."""

    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class MambaBlock(nn.Module):
    """Mamba-based residual block used by :class:`MambaDenoiser`.

    Each row is treated as a length-1 sequence (``x.unsqueeze(1)``): Mamba's
    conv1d/SSM scan run over that single timestep, so this uses Mamba as a
    gated nonlinear feature-mixing layer rather than for genuine sequence
    modeling (tabular rows have no inherent order) -- this matches the
    upstream reference implementation exactly.
    """

    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.1):
        super().__init__()
        from mamba_ssm import Mamba

        self.norm = nn.LayerNorm(dim_in)
        self.mamba = Mamba(dim_in, d_conv=2, expand=1)
        self.proj = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        x = self.norm(x)
        x = self.mamba(x.unsqueeze(1)).squeeze(1)
        x = self.proj(x)
        x = self.dropout(x)
        return x + residual


class MambaDenoiser(nn.Module):
    """EDM denoiser network with a Mamba-based diamond up/down-sampling body.

    Ported from RefiDiff's ``model.py::CustomDenoiser``.
    """

    def __init__(self, d_in: int, dim_t: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, dim_t)
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))
        self.up1 = MambaBlock(dim_t, dim_t * 2, dropout)
        self.up2 = MambaBlock(dim_t * 2, dim_t * 4, dropout)
        self.down1 = MambaBlock(dim_t * 4, dim_t * 2, dropout)
        self.down2 = MambaBlock(dim_t * 2, dim_t, dropout)
        self.output_proj = nn.Linear(dim_t, d_in)

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        x = self.input_proj(x) + emb
        x = self.up1(x)
        x = self.up2(x)
        x = self.down1(x)
        x = self.down2(x)
        return self.output_proj(x)


class MlpResidualBlock(nn.Module):
    """Plain residual MLP block, structurally analogous to :class:`MambaBlock`."""

    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.linear = nn.Linear(dim_in, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        x = self.norm(x)
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class MlpResidualDenoiser(nn.Module):
    """Fallback EDM denoiser used when mamba-ssm is unavailable.

    Mirrors :class:`MambaDenoiser`'s diamond up/down-sampling shape
    (``dim_t -> 2x -> 4x -> 2x -> dim_t``) with plain ``nn.Linear`` +
    residual blocks in place of Mamba. The paper's own ablation (Appendix J)
    reports only ~2% performance difference between the two, so this is a
    reasonable CPU-only / no-mamba-ssm fallback, not just a stub.
    """

    def __init__(self, d_in: int, dim_t: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, dim_t)
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))
        self.up1 = MlpResidualBlock(dim_t, dim_t * 2, dropout)
        self.up2 = MlpResidualBlock(dim_t * 2, dim_t * 4, dropout)
        self.down1 = MlpResidualBlock(dim_t * 4, dim_t * 2, dropout)
        self.down2 = MlpResidualBlock(dim_t * 2, dim_t, dropout)
        self.output_proj = nn.Linear(dim_t, d_in)

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        x = self.input_proj(x) + emb
        x = self.up1(x)
        x = self.up2(x)
        x = self.down1(x)
        x = self.down2(x)
        return self.output_proj(x)


def _resolve_denoiser_cls(denoiser: str) -> tuple:
    """Resolve ``RefiDiffConfig.denoiser`` ("auto"/"mamba"/"mlp") to a denoiser class.

    See /memories/repo/refidiff-notes.md for the specific mamba-ssm/causal-conv1d
    version pins confirmed to build+run on this machine (do not let the
    `refidiff` extra float to "latest" -- newer releases force a disruptive
    torch/CUDA upgrade).
    """
    if denoiser == "mlp":
        logger.info("refidiff: denoiser='mlp' -- using MlpResidualDenoiser")
        return MlpResidualDenoiser, "mlp"
    if denoiser == "mamba":
        try:
            import mamba_ssm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "imputation.refidiff.denoiser='mamba' requires the mamba-ssm package. "
                "Install it via `uv sync --extra refidiff` (see "
                "/memories/repo/refidiff-notes.md for the pinned working versions), "
                "or set imputation.refidiff.denoiser to 'auto' or 'mlp' to use the "
                "MLP fallback instead."
            ) from exc
        logger.info("refidiff: denoiser='mamba' -- using MambaDenoiser")
        return MambaDenoiser, "mamba"
    # "auto"
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        logger.info(
            "refidiff: denoiser='auto' and mamba-ssm is not importable -- falling back to "
            "MlpResidualDenoiser. Install the `refidiff` extra for the Mamba denoiser."
        )
        return MlpResidualDenoiser, "mlp"
    logger.info("refidiff: denoiser='auto' -- mamba-ssm is importable, using MambaDenoiser")
    return MambaDenoiser, "mamba"


# ---------------------------------------------------------------------------
# EDM diffusion preconditioning, loss, and model (ported from model.py)
# ---------------------------------------------------------------------------


class Precond(nn.Module):
    """EDM preconditioning wrapper around a denoiser network (ported from model.py::Precond)."""

    def __init__(
        self,
        denoise_fn: nn.Module,
        sigma_min: float = 0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        f_x = self.denoise_fn_F(x=(c_in * x).to(torch.float32), noise_labels=c_noise.flatten())
        return c_skip * x + c_out * f_x.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class EDMLoss:
    """EDM training loss (ported from diffusion_utils.py::EDMLoss)."""

    def __init__(self, p_mean: float = -1.2, p_std: float = 1.2, sigma_data: float = 0.5):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data

    def __call__(self, denoise_fn: Precond, data: torch.Tensor) -> torch.Tensor:
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(data) * sigma.unsqueeze(1)
        d_yn = denoise_fn(data + noise, sigma)
        return weight.unsqueeze(1) * ((d_yn - data) ** 2)


class Model(nn.Module):
    """Thin wrapper tying a denoiser to :class:`Precond` + :class:`EDMLoss` (ported from model.py::Model)."""

    def __init__(self, denoise_fn: nn.Module):
        super().__init__()
        self.denoise_fn_D = Precond(denoise_fn)
        self.loss_fn = EDMLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self.denoise_fn_D, x).mean(-1).mean()


# ---------------------------------------------------------------------------
# Reverse-diffusion sampling with observed-value clamping (ported from
# diffusion_utils.py::impute_mask)
# ---------------------------------------------------------------------------


def _sample_step(
    net: Precond,
    num_steps: int,
    i: int,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    x_next: torch.Tensor,
) -> torch.Tensor:
    x_cur = x_next
    gamma = min(_S_CHURN / num_steps, math.sqrt(2) - 1) if _S_MIN <= t_cur <= _S_MAX else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * _S_NOISE * torch.randn_like(x_cur)

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def _impute_mask(
    net: Precond,
    x: torch.Tensor,
    missing_mask: torch.Tensor,
    num_steps: int,
    device: str,
) -> torch.Tensor:
    """One reverse-diffusion trajectory: observed entries stay clamped near ``x``.

    ``missing_mask`` uses the convention 1 == missing (imputed via the
    diffusion model), 0 == observed (kept close to ``x`` plus schedule
    noise), consistently with :func:`_warmup_refine`'s ``missing_mask``.
    """
    num_samples, dim = x.shape
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min = max(_SIGMA_MIN, net.sigma_min)
    sigma_max = min(_SIGMA_MAX, net.sigma_max)
    t_steps = (
        sigma_max ** (1 / _RHO)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / _RHO) - sigma_max ** (1 / _RHO))
    ) ** _RHO
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    mask = missing_mask.to(torch.float32).to(device)
    x_t = torch.randn([num_samples, dim], device=device).to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:], strict=True)):
            if i >= num_steps - 1:
                continue
            for j in range(_N_LANGEVIN):
                n_prev = torch.randn_like(x_t) * t_next
                x_known_t_prev = x + n_prev
                x_unknown_t_prev = _sample_step(net, num_steps, i, t_cur, t_next, x_t)
                x_t_prev = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask
                if j == _N_LANGEVIN - 1:
                    x_t = x_t_prev
                else:
                    noise = torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt()
                    x_t = x_t_prev + noise
    return x_t


# ---------------------------------------------------------------------------
# Training loop with checkpointing/resumability
# ---------------------------------------------------------------------------


def _config_hash(in_dim: int, cfg: RefiDiffConfig, denoiser_name: str) -> str:
    payload = json.dumps(
        {"in_dim": in_dim, "hidden_dim": cfg.hidden_dim, "denoiser": denoiser_name},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _checkpoint_dir(data_dir: Path, in_dim: int, cfg: RefiDiffConfig, denoiser_name: str) -> Path:
    return Path(data_dir) / ".refidiff_checkpoints" / _config_hash(in_dim, cfg, denoiser_name)


def _train(
    model: Model,
    train_data: torch.Tensor,
    device: str,
    cfg: RefiDiffConfig,
    checkpoint_dir: Path,
) -> Model:
    """Train ``model`` with Adam + ReduceLROnPlateau + early stopping.

    Checkpoints (model/optimizer/scheduler state + epoch/best_loss/patience)
    are written every ``cfg.checkpoint_every`` epochs so an interrupted run
    (e.g. shared-GPU preemption/OOM) resumes instead of retraining from
    scratch -- constitution principle 6 (long-running jobs must checkpoint
    and resume).
    """
    ensure_dir(checkpoint_dir)
    best_model_path = checkpoint_dir / "best_model.pt"
    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=40
    )

    start_epoch = 0
    best_loss = float("inf")
    patience_count = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        patience_count = checkpoint["patience_count"]
        logger.info(
            "refidiff: resuming training from checkpoint epoch %d (best_loss=%.4f, "
            "patience=%d) at %s",
            start_epoch,
            best_loss,
            patience_count,
            checkpoint_dir,
        )
    elif best_model_path.exists():
        logger.info(
            "refidiff: found a completed run's best_model.pt at %s (no in-progress "
            "checkpoint) -- reusing it instead of retraining",
            checkpoint_dir,
        )
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        return model

    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=min(cfg.batch_size, len(train_dataset)), shuffle=True
    )

    model.train()
    epoch_iter = tqdm(
        range(start_epoch, cfg.epochs), desc="refidiff training", unit="epoch", initial=start_epoch
    )
    for epoch in epoch_iter:
        batch_loss = 0.0
        n_seen = 0
        for (batch,) in train_loader:
            batch = batch.float().to(device)
            loss = model(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item() * len(batch)
            n_seen += len(batch)
        curr_loss = batch_loss / max(n_seen, 1)
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience_count = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_count += 1

        epoch_iter.set_postfix(
            loss=f"{curr_loss:.4f}",
            best=f"{best_loss:.4f}",
            patience=f"{patience_count}/{cfg.early_stopping_patience}",
            refresh=False,
        )

        if epoch % cfg.checkpoint_every == 0 or patience_count >= cfg.early_stopping_patience:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "patience_count": patience_count,
                },
                checkpoint_path,
            )

        if patience_count >= cfg.early_stopping_patience:
            logger.info("refidiff: early stopping at epoch %d (best_loss=%.4f)", epoch, best_loss)
            break

    # Training finished (converged or hit max epochs): load the best
    # snapshot for inference and drop the resumability checkpoint so a
    # future call with an identical config starts a fresh run rather than
    # "resuming" one that already finished.
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return model


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def impute_dataframe(
    df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    device: str,
    refidiff_cfg: RefiDiffConfig,
    data_dir: Path,
) -> pd.DataFrame:
    """Impute missing values in ``feature_columns`` of ``df`` via RefiDiff.

    The target column is assumed fully observed and is passed through
    unchanged. Returns a new DataFrame with the same column order as ``df``.
    Training checkpoints are stored under
    ``data_dir/.refidiff_checkpoints/<config-hash>/``.
    """
    cfg = refidiff_cfg
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]
    n_samples = len(df)

    logger.info(
        "refidiff: imputing %d rows, %d numeric + %d categorical feature columns on device=%s",
        n_samples,
        len(numeric_columns),
        len(categorical_columns),
        device,
    )

    # --- Encode: numeric columns first, then binary-encoded categorical
    # bit-columns (matches RefiDiff's `col >= len_num` convention). ---
    # A column not listed in categorical_columns is assumed to already be
    # numeric (e.g. an ordinal column pre-encoded to integers), but a plain
    # CSV source can still surprise us with a string-valued "numeric" column
    # (e.g. an ordinal band stored as text like "Light"/"Heavy"). Label-encode
    # any such column as a fallback -- mirrors tabimpute_backend's identical
    # use of this helper, so both backends handle the same input the same way.
    numeric_df, numeric_category_maps = label_encode_non_numeric_columns(df, numeric_columns)
    if numeric_category_maps:
        logger.warning(
            "refidiff: %d feature column(s) not listed in data.categorical_columns contain "
            "non-numeric values and were label-encoded as a fallback: %s. This preserves "
            "missingness correctly but does not guarantee categories are numbered in their "
            "true ordinal order (alphabetical by default) -- add them to "
            "data.categorical_columns (if nominal), or to an explicit ordinal-to-integer "
            "mapping at the data-loading stage (if ordinal), for more correct treatment.",
            len(numeric_category_maps),
            sorted(numeric_category_maps),
        )
    if numeric_columns:
        numeric_missing = numeric_df.isna().to_numpy()
        numeric_values = np.nan_to_num(numeric_df.to_numpy(dtype=np.float64), nan=0.0)
    else:
        numeric_missing = np.zeros((n_samples, 0), dtype=bool)
        numeric_values = np.zeros((n_samples, 0))

    cat_encoders = _fit_categorical_binary_encoders(df, categorical_columns)
    cat_bit_blocks, cat_missing_blocks, cat_bin_widths = [], [], []
    for col in categorical_columns:
        bits, missing = _encode_categorical_to_bits(df[col], cat_encoders[col])
        cat_bit_blocks.append(bits)
        cat_missing_blocks.append(np.repeat(missing[:, None], bits.shape[1], axis=1))
        cat_bin_widths.append(bits.shape[1])

    cat_values = (
        np.concatenate(cat_bit_blocks, axis=1) if cat_bit_blocks else np.zeros((n_samples, 0))
    )
    cat_missing = (
        np.concatenate(cat_missing_blocks, axis=1)
        if cat_missing_blocks
        else np.zeros((n_samples, 0), dtype=bool)
    )

    len_num = len(numeric_columns)
    x_raw = np.concatenate([numeric_values, cat_values], axis=1)
    missing_mask = np.concatenate([numeric_missing, cat_missing], axis=1)

    logger.info(
        "refidiff: encoded feature matrix shape=%s (%d numeric + %d binary-encoded categorical "
        "bit columns), %d/%d missing entries",
        x_raw.shape,
        len_num,
        cat_values.shape[1],
        int(missing_mask.sum()),
        missing_mask.size,
    )

    # --- Standardize (z-score using observed entries only), then /2 to
    # match RefiDiff's own diffusion input scale (main.py: (X-mean)/std/2). ---
    mean, std = _mean_std(x_raw, missing_mask)
    x = (x_raw - mean) / std / 2.0

    # --- Warm-up refinement: single-pass per-column fill. ---
    logger.info("refidiff: running warm-up refinement pass")
    x_warm = x.copy()
    x_warm[missing_mask] = 0.0
    x_warm = _warmup_refine(x_warm, missing_mask, len_num, device=device)

    # --- Train the EDM diffusion model on the warm-up-filled data. ---
    in_dim = x_warm.shape[1]
    denoiser_cls, denoiser_name = _resolve_denoiser_cls(cfg.denoiser)
    denoise_fn = denoiser_cls(in_dim, cfg.hidden_dim).to(device)
    model = Model(denoise_fn=denoise_fn).to(device)
    n_params = sum(p.numel() for p in denoise_fn.parameters())
    logger.info(
        "refidiff: training %s denoiser (in_dim=%d, hidden_dim=%d, %d params) for up to "
        "%d epochs (early_stopping_patience=%d)",
        denoiser_name,
        in_dim,
        cfg.hidden_dim,
        n_params,
        cfg.epochs,
        cfg.early_stopping_patience,
    )

    train_data = torch.tensor(x_warm, dtype=torch.float32)
    checkpoint_dir = _checkpoint_dir(data_dir, in_dim, cfg, denoiser_name)
    model = _train(model, train_data, device, cfg, checkpoint_dir)

    # --- Reverse-diffusion sampling: average num_trials trajectories. ---
    model.eval()
    net = model.denoise_fn_D
    x_known = torch.tensor(x_warm, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(missing_mask, dtype=torch.float32, device=device)

    logger.info(
        "refidiff: sampling %d reverse-diffusion trajectories (%d steps each)",
        cfg.num_trials,
        cfg.num_steps,
    )
    trial_results = []
    for _trial in tqdm(range(cfg.num_trials), desc="refidiff sampling", unit="trial"):
        rec = _impute_mask(net, x_known, mask_tensor, cfg.num_steps, device)
        rec = rec * mask_tensor + x_known * (1 - mask_tensor)
        trial_results.append(rec)
    rec_x = torch.stack(trial_results, dim=0).mean(0).detach().cpu().numpy()

    # --- Polishing pass: re-run warm-up refinement on the diffusion output. ---
    logger.info("refidiff: running polishing refinement pass")
    rec_x = _warmup_refine(rec_x, missing_mask, len_num, device=device)

    # --- De-standardize back to raw feature units. ---
    rec_x = rec_x * 2.0 * std + mean

    # --- Reassemble into the original DataFrame layout. ---
    # Build numeric and categorical columns as plain dicts first and construct
    # each block's DataFrame in one shot, then concat once. Assigning columns
    # one at a time onto a single growing DataFrame (result[col] = ...) forces
    # a `frame.insert` per column, which for ~1000+ columns triggers pandas'
    # "DataFrame is highly fragmented" PerformanceWarning repeatedly.
    numeric_result = pd.DataFrame(
        {col: rec_x[:, i] for i, col in enumerate(numeric_columns)}, index=df.index
    )
    numeric_result = decode_label_encoded_columns(numeric_result, numeric_category_maps)

    categorical_data = {}
    offset = len_num
    for col, width in zip(categorical_columns, cat_bin_widths, strict=True):
        bits = rec_x[:, offset : offset + width]
        categorical_data[col] = _decode_bits_to_categorical(bits, cat_encoders[col], col)
        offset += width
    categorical_result = pd.DataFrame(categorical_data, index=df.index)

    result = pd.concat([numeric_result, categorical_result], axis=1)
    result[target_column] = df[target_column].values
    return result[list(df.columns)]
