"""Regenerate the committed integration-test fixture (``clinic.csv``).

The fixture is a small, fully synthetic clinical-style table with a known
data-generating process, so integration tests can assert on real pipeline
outputs without network access or private data. Run from the repository root:

    uv run python tests/integration/fixtures/make_fixture.py

The output is deterministic for a given numpy version; commit the CSV, not
just this script, so tests never depend on regeneration.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261007
N_PATIENTS = 160
#: Share of patients with a second encounter (exercises patient-group splitting).
REPEAT_FRACTION = 0.25
#: MCAR missingness injected into these feature columns only (never the target).
MISSING_RATES = {"BMI": 0.10, "LAB_A": 0.10, "SEVERITY": 0.08, "SITE": 0.06}


def make_fixture(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    patients = pd.DataFrame(
        {
            "patient_id": np.arange(1000, 1000 + N_PATIENTS),
            "AGE": rng.integers(18, 80, size=N_PATIENTS),
            "SEX": rng.choice(["F", "M"], size=N_PATIENTS),
            "SITE": rng.choice(["north", "south", "east"], size=N_PATIENTS, p=[0.4, 0.35, 0.25]),
            "SMOKER": rng.choice([0, 1], size=N_PATIENTS, p=[0.7, 0.3]),
            "DIAGNOSIS": rng.choice(["anxiety", "depression", "none"], size=N_PATIENTS),
        }
    )
    repeats = patients.sample(frac=REPEAT_FRACTION, random_state=seed)
    rows = pd.concat([patients, repeats], ignore_index=True)
    n = len(rows)

    # Encounter-level measurements with known dependencies on patient traits.
    age = rows["AGE"].to_numpy(dtype=float)
    smoker = rows["SMOKER"].to_numpy(dtype=float)
    male = (rows["SEX"] == "M").to_numpy(dtype=float)
    rows["BMI"] = np.round(22 + 0.06 * (age - 45) + 1.5 * male + rng.normal(0, 3, n), 1)
    rows["LAB_A"] = np.round(1.0 + 0.8 * smoker + 0.01 * age + rng.normal(0, 0.4, n), 2)
    rows["LAB_B"] = np.round(50 + 0.5 * rows["BMI"] + rng.normal(0, 5, n), 1)
    severity_score = 0.03 * (age - 45) + 0.9 * smoker + 0.6 * (rows["LAB_A"] - 1.5)
    rows["SEVERITY"] = np.digitize(severity_score + rng.normal(0, 0.5, n), [-0.3, 0.4, 1.1])

    # Binary outcome from a logistic model, so a classifier trained on good
    # synthetic data has real signal to recover (TSTR well above chance).
    logit = -0.5 + 0.9 * rows["SEVERITY"] + 0.6 * smoker + 0.03 * (rows["BMI"] - 24)
    rows["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    for column, rate in MISSING_RATES.items():
        mask = rng.random(n) < rate
        rows[column] = rows[column].astype(object if column == "SITE" else float)
        rows.loc[mask, column] = np.nan

    rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    columns = [
        "patient_id",
        "AGE",
        "SEX",
        "SITE",
        "SMOKER",
        "BMI",
        "LAB_A",
        "LAB_B",
        "SEVERITY",
        "DIAGNOSIS",
        "target",
    ]
    return rows[columns]


if __name__ == "__main__":
    out = Path(__file__).with_name("clinic.csv")
    make_fixture().to_csv(out, index=False)
    print(f"wrote {out}")
