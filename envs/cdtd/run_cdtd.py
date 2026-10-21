"""Fit CDTD on one table and sample from it.

Runs inside envs/cdtd (its own environment, see pyproject.toml); called by
synthdata.generation.cdtd_backend, which writes the inputs and reads the
output. It imports nothing from synthdata. The model code is the authors'
cdtd_simple (``spec["source_dir"]``, the submodules/cdtd_simple checkout).

    python run_cdtd.py --train train.parquet --spec spec.json --out synthetic.parquet

Preprocessing follows cdtd_simple's main.ipynb: categories as strings, then
integer codes (OrdinalEncoder); continuous columns through a normal-output
QuantileTransformer and a StandardScaler, inverted after sampling. The one
addition is seeding the model's weight initialisation (see below).

spec.json keys: ``source_dir``, ``categorical`` and ``continuous`` (column
names), ``params`` (CDTD() arguments), ``fit_params`` (CDTD.fit() arguments),
``sample_params`` (CDTD.sample() arguments), ``seed`` and ``n_samples``.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    sys.path.insert(0, spec["source_dir"])
    from cdtd import CDTD
    from cdtd.utils import set_seeds

    torch.set_float32_matmul_precision("high")  # as in main.ipynb
    train = pd.read_parquet(args.train)
    cat_cols, cont_cols = spec["categorical"], spec["continuous"]
    seed = spec["seed"]

    cat_enc = OrdinalEncoder()
    x_cat = torch.tensor(cat_enc.fit_transform(train[cat_cols].to_numpy().astype(str))).long()
    x_cont_raw = train[cont_cols].to_numpy().astype(float)
    cont_q = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=max(min(x_cont_raw.shape[0] // 30, 1000), 10),
        subsample=int(1e9),
        random_state=seed,
    )
    cont_std = StandardScaler()
    x_cont = torch.tensor(cont_std.fit_transform(cont_q.fit_transform(x_cont_raw))).float()

    # cdtd_simple seeds fit() and sample() but not the weight initialisation in
    # CDTD(), so two runs with one seed differ; seed it too.
    set_seeds(seed)
    model = CDTD(x_cat, x_cont, **spec["params"])
    fit_params = {"batch_size": 4096, **spec["fit_params"]}
    # cdtd_simple's loader drops the last partial batch, so a batch larger
    # than the table would yield no batches and training would never end.
    fit_params["batch_size"] = min(fit_params["batch_size"], len(train))
    model.fit(x_cat, x_cont, seed=seed, **fit_params)
    gen_cat, gen_cont = model.sample(spec["n_samples"], seed=seed, **spec["sample_params"])

    out = pd.DataFrame(cat_enc.inverse_transform(gen_cat), columns=cat_cols)
    cont = cont_q.inverse_transform(cont_std.inverse_transform(gen_cont))
    for i, col in enumerate(cont_cols):
        out[col] = cont[:, i]
    out = out[[c for c in train.columns if c in out]]
    out[cat_cols] = out[cat_cols].astype(str)
    out.astype({c: np.float64 for c in cont_cols}).to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
