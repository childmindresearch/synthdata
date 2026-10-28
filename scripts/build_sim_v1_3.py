"""Build the simulated clinical dataset 1.3 from 1.2.

Reads ``sim_v1.2.csv`` and ``sim_v1.2_variable_schema.csv`` from
``--data-dir`` and writes ``sim_v1.3.csv`` and ``sim_v1.3_variable_schema.csv``
next to them; 1.2 is left untouched. The run is deterministic (fixed seed),
so any machine holding 1.2 rebuilds the same 1.3.

Two passes, in this order:

1. **Dependencies the 1.2 simulation skipped**, enforced on the raw 1.2
   columns. Each rule changes the dependent column as little as it can and
   the change log counts the rows it changed.
2. **Consolidation**: drop redundant or too granular columns, group labs,
   drug screens, medication classes and income sources, turn labs into
   missing/low/normal/high, merge rare levels, recode "not applicable"
   codes, then write the schema (types and ordinal orders).

The change log (stdout) lists the columns dropped, added and recoded and the
rows each rule changed. It prints column names, level names, counts and
aggregate statistics only, never row values.

Example::

    uv run python scripts/build_sim_v1_3.py --data-dir data/simulated
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261028
#: Levels with fewer rows than this are merged into "Other".
MIN_LEVEL_ROWS = 20
ID_COLUMNS = ["encounter_number", "research_subject_id"]

#: Adult reference ranges (SI units) as ``(low, high)``; a value below
#: ``low`` is "low", above ``high`` is "high". Sex-specific ranges are
#: ``{"female": (...), "male": (...)}``; other or unknown genders use the
#: union of the two. ``None`` uses the central 95% of the observed values
#: (how reference intervals are defined) because the 1.2 values are not on a
#: clinical scale. Drug levels use their therapeutic ranges.
LAB_RANGES: dict[str, tuple[float, float] | dict[str, tuple[float, float]] | None] = {
    "WBC": (4.0, 11.0),  # 10^9/L
    "neutrophil": (2.0, 7.5),  # 10^9/L
    "hgb": {"female": (120.0, 150.0), "male": (130.0, 170.0)},  # g/L
    "platelet": (150.0, 400.0),  # 10^9/L
    "sodium": (135.0, 145.0),  # mmol/L
    "potassium": (3.5, 5.0),  # mmol/L
    "creatinine": {"female": (45.0, 90.0), "male": (60.0, 110.0)},  # umol/L
    "glucose": (3.9, 7.8),  # mmol/L, random (non-fasting) upper limit
    "Hb A1c": (0.04, 0.06),  # fraction (4-6%)
    "HDL": None,  # median about 9x the clinical mmol/L scale
    "LDL": (0.0, 3.4),  # mmol/L
    "cholesterol": (0.0, 5.2),  # mmol/L
    "triglyceride": (0.0, 1.7),  # mmol/L
    "ALT": (7.0, 56.0),  # U/L
    "AST": (10.0, 40.0),  # U/L
    "GGT": {"female": (5.0, 36.0), "male": (8.0, 61.0)},  # U/L
    "albumin": (35.0, 50.0),  # g/L
    "TSH": (0.4, 4.0),  # mIU/L
    "CRP": (0.0, 10.0),  # mg/L
    "B12": (148.0, 664.0),  # pmol/L
    "clozapine": (350.0, 600.0),  # ug/L, therapeutic
    "valproic acid": (50.0, 100.0),  # mg/L, therapeutic
}
#: Analytes dropped: derivable from kept ones (non-HDL = cholesterol - HDL,
#: eGFR from creatinine) or redundant with hgb (hct, RBC).
DROPPED_LABS = ["hct", "RBC", "non-HDL", "eGFR"]
#: One count per panel (the max of its members' counts), replacing theirs.
LAB_PANELS = {
    "cbc_count": ["WBC", "neutrophil", "hgb", "hct", "platelet", "RBC"],
    "renal_count": ["sodium", "potassium", "creatinine", "eGFR"],
    "lipid_count": ["HDL", "LDL", "cholesterol", "triglyceride"],
    "liver_count": ["ALT", "AST", "GGT", "albumin"],
}
LAB_LEVELS = ["missing", "low", "normal", "high"]

#: Drug screens. Every screened drug has ``<drug>_neg`` and ``<drug>_pos``
#: counts; screens run = neg + pos. In 1.2 the per-drug totals are
#: independent (pairwise correlation near 0), so no panel can be read off
#: the data; the urine panel follows the usual immunoassay panel, and
#: cannabis and antipsychotic levels are counted apart.
UDS_DRUGS = [
    "cocaine",
    "benzodia",
    "opiate",
    "fentanyl",
    "oxycodone",
    "EDDP",
    "buprenorphine",
    "methadone",
    "ethanol",
]
AP_DRUGS = ["olanzapine", "risperidone", "quetiapine", "loxapine", "ziprasidone"]
SCREEN_COUNTS = {"n_uds": UDS_DRUGS, "n_cannab_screens": ["cannab"], "n_ap_screens": AP_DRUGS}
POSITIVE_GROUPS = {
    "opioid_pos_any": ["opiate", "fentanyl", "oxycodone"],
    "oat_pos_any": ["methadone", "EDDP", "buprenorphine"],
    "cocaine_pos_any": ["cocaine"],
    "benzodia_pos_any": ["benzodia"],
    "cannab_pos_any": ["cannab"],
    "ethanol_pos_any": ["ethanol"],
    "ap_level_detected_any": AP_DRUGS,
}

#: Medication classes kept as their own 0/1 flag.
MED_CORE = [
    "Antipsychotic",
    "Antidepressant",
    "Mood stabilizer",
    "Anticonvulsant (psych)",
    "Benzodiazepine",
    "ADHD/stimulant",
    "Anticholinergic (EPS)",
    "Clonidine (alpha-2 agonist)",
    "OAT",
    "SUD treatment",
]
#: The other classes, grouped into 0/1 "any" flags.
MED_GROUPS = {
    "med_sedative_other": ["Z-drug", "Non-benzo anxiolytic"],
    "med_anti_infective": [
        "Antibiotic",
        "Antifungal",
        "Antimycobacterial",
        "Antiretroviral",
        "Antiviral",
        "Hepatitis treatment",
        "Gut-restricted antibiotic",
    ],
    "med_gi": ["Acid suppression", "Antidiarrheal", "Antiemetic", "Laxative"],
    "med_cardiometabolic": ["Cardiovascular", "Lipid-lowering", "Antidiabetic", "Antithrombotic"],
    "med_endocrine": ["Thyroid", "Hormonal", "Bone/endocrine", "Anti-androgen", "Corticosteroid"],
    "med_neuro": [
        "Anticonvulsant (neuro)",
        "Parkinson's",
        "Triptan",
        "CGRP/other migraine",
        "Cognitive enhancer",
    ],
    "med_other_somatic": [
        "Anti-gout",
        "Antihistamine",
        "Immunosuppressant/biologic",
        "Iron",
        "NSAID",
        "Nasal preparation",
        "Opioid analgesic",
        "Overactive bladder",
        "Respiratory",
    ],
}

INCOME_FLAGS = {
    "employment": ["Income - Employment"],
    "government": [
        "Income - Employment Insurance",
        "Income - Social Assistance",
        "Income - Disability Insurance",
        "Income - Pension",
    ],
    "other": ["Income - Other"],
}
NO_INCOME = "Income - No Income"

#: Columns whose code 8 means "did not occur" or "not applicable", outside
#: the item's scale; it becomes missing. Education (codes 1-6, 8, 9) and
#: Acute Control Medications (0-9, a count) are left alone: there 8 is a
#: level on the scale.
NOT_APPLICABLE_8 = [
    "ADL - Personal Hygiene",
    "ADL - Walking",
    "ADL - Wheeling",
    "Self-Injury - Intent",
    "Life Event (J1) Causes Sense of Horror or Intense Fear",
    "History of Medication Adherence",
    "Time Since Last Discharge",
    "Risk of Unemployment/Disrupted Education - Poor Productivity/Disruptiveness at School/Work",
    "Risk of Unemployment/Disrupted Education - Persistent Unemployment",
    "Risk of Unemployment/Disrupted Education - Increase in Lateness/Absenteeism",
    "Risk of Unemployment/Disrupted Education - Intent to Quit Work/School",
]
#: Ordinal orders added in 1.3 (lowest to highest), after the code-8 recode.
NEW_ORDINAL_ORDERS: dict[str, list] = {
    "ADL - Personal Hygiene": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "ADL - Walking": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "ADL - Wheeling": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "Time Since Last Discharge": [1.0, 2.0, 3.0, 4.0],
    "Self-Injury - Intent": [0.0, 1.0],
    "History of Medication Adherence": [0.0, 1.0, 2.0, 3.0],
    **{
        c: [0.0, 1.0]
        for c in NOT_APPLICABLE_8
        if c.startswith("Risk of Unemployment/Disrupted Education")
    },
    "Discharge Readiness - Patient Expresses Preference to Return to Community": [False, True],
    "Discharge Readiness - Patient has Support for Discharge": [False, True],
}
RARE_LEVEL_COLUMNS = [
    "region",
    "lhin",
    "encounter_hospitalization_admit_source",
    "encounter_hospitalization_discharge_disposition",
]
#: Duplicate sources: (kept, dropped).
DUPLICATE_SOURCES = [
    ("encounter_hospitalization_admit_source", "Admitted From"),
    ("encounter_hospitalization_discharge_disposition", "Discharged To"),
    ("encounter_hospitalization_discharge_disposition", "Discharge Reason"),
]
DROPPED_DEMOGRAPHICS = [
    "encounter_end_year",
    # age_reg already equals encounter_start_year - birth_year (and has no
    # blanks), so it is the age at admission and birth_year adds nothing.
    "birth_year",
    "encounter_location_name",  # too granular to release
    "Born in Canada",
    "Canadian citizenship",
    "Speak English",
    "Continent of Birth",
]


class ChangeLog:
    """Collects the change log and prints it at the end."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, title: str) -> None:
        self.lines.append(f"\n== {title}")

    def note(self, text: str) -> None:
        self.lines.append(f"- {text}")

    def print(self) -> None:
        print("\n".join(self.lines))


def changed(before: pd.Series, after: pd.Series) -> int:
    """Rows whose value differs (two missing values count as equal)."""
    return int(((before != after) & ~(before.isna() & after.isna())).sum())


def nest(df: pd.DataFrame, child: str, parent: str, log: ChangeLog) -> None:
    """Make ``child`` nest in ``parent``: each child level maps to one parent.

    Each child level maps to the parent it appears with most often, except
    that every parent level keeps at least one child: in 1.2 the two columns
    are nearly independent, so the most common parent is the same one or two
    levels for every child. Parents are covered first, greedily by
    co-occurrence count. The parent is then set from the child wherever the
    child is known, which also fills blank parents.
    """
    both = df[[child, parent]].dropna()
    pairs = both.groupby([child, parent]).size().sort_values(ascending=False, kind="stable")
    assignment: dict = {}
    for c, p in pairs.index:
        if c not in assignment and p not in assignment.values():
            assignment[c] = p
    for c, p in pairs.index:
        assignment.setdefault(c, p)
    mapping = pd.Series(assignment)
    before = df[parent].copy()
    known = df[child].notna()
    df.loc[known, parent] = df.loc[known, child].map(mapping)
    filled = int((before.isna() & df[parent].notna()).sum())
    log.note(
        f"{child} nests in {parent}: {mapping.size} {child} levels mapped to "
        f"{mapping.nunique()} {parent} levels; {changed(before, df[parent]) - filled} rows "
        f"changed {parent}, {filled} blank {parent} filled"
    )


def raise_to(df: pd.DataFrame, mask: pd.Series, column: str, minimum: float) -> int:
    """Set ``column`` to at least ``minimum`` on ``mask`` rows; return rows changed."""
    before = df[column].copy()
    rows = mask & ~(df[column] >= minimum)
    df.loc[rows, column] = minimum
    return changed(before, df[column])


def sample_nonzero(df: pd.DataFrame, column: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` codes from ``column``'s observed nonzero distribution."""
    counts = df.loc[df[column] > 0, column].value_counts()
    return rng.choice(counts.index.to_numpy(), size=n, p=(counts / counts.sum()).to_numpy())


def enforce_dependencies(df: pd.DataFrame, rng: np.random.Generator, log: ChangeLog) -> None:
    """Apply the hard rules the 1.2 simulation skipped, on the raw columns."""
    log.section("A. Dependencies (raw 1.2 columns)")
    nest(df, "sub_program", "major_program", log)
    small = df["sub_program"].value_counts()
    small = small[small < MIN_LEVEL_ROWS].index
    if len(small):
        rows = df["sub_program"].isin(small)
        df.loc[rows, "sub_program"] = df.loc[rows, "major_program"] + " - Other"
    log.note(f"sub_program levels under {MIN_LEVEL_ROWS} rows merged: {len(small)}")
    nest(df, "lhin", "region", log)

    log.note(
        "Schizophrenia Primary Diagnosis = 1 => Schizophrenia spectrum disorder: "
        f"{raise_to(df, df['Schizophrenia Primary Diagnosis'] == 1, 'Schizophrenia spectrum disorder', True)}"
        " rows changed"
    )

    clozapine = df["clozapine_count"] > 0
    log.note(
        "clozapine level measured => Treatment with Clozapine = 1: "
        f"{raise_to(df, clozapine, 'Treatment with Clozapine', 1.0)} rows changed"
    )
    ap_detected = (df[[f"{d}_pos" for d in AP_DRUGS]] > 0).any(axis=1)
    for name, mask in [
        ("clozapine level measured", clozapine),
        ("Treatment with Clozapine = 1", df["Treatment with Clozapine"] == 1),
        ("Long Acting Injectable Medication >= 1", df["Long Acting Injectable Medication"] >= 1),
        ("antipsychotic level detected on a screen", ap_detected),
    ]:
        log.note(
            f"{name} => Antipsychotic >= 1: {raise_to(df, mask, 'Antipsychotic', 1.0)} rows changed"
        )
    log.note(
        "lithium level measured => Mood stabilizer = 1: "
        f"{raise_to(df, df['lithium_count'] > 0, 'Mood stabilizer', 1.0)} rows changed"
    )

    # Suicide plan => self-injury considered (codes 1-5 are how recently;
    # 0 is never). A missing "considered" code is drawn from the observed
    # nonzero codes of rows with a plan.
    rows = (df["Suicide Plan"] == 1) & (df["Self-Injury - Considered"] == 0)
    with_plan = df[df["Suicide Plan"] == 1]
    df.loc[rows, "Self-Injury - Considered"] = sample_nonzero(
        with_plan, "Self-Injury - Considered", int(rows.sum()), rng
    )
    log.note(f"Suicide Plan = 1 => Self-Injury - Considered > 0: {int(rows.sum())} rows changed")

    # Horror or fear caused by a life event (J1 = 1) => at least one life
    # event happened (codes 1-5 are how recently; 0 is never). One observed
    # item, chosen at random, gets a code drawn from its nonzero codes.
    events = [c for c in df.columns if c.startswith("Life Events and History - ")]
    observed = df[events].notna()
    rows = (
        (df["Life Event (J1) Causes Sense of Horror or Intense Fear"] == 1)
        & observed.any(axis=1)
        & ~(df[events] > 0).any(axis=1)
    )
    for idx in df.index[rows]:
        item = rng.choice([c for c in events if observed.at[idx, c]])
        df.at[idx, item] = sample_nonzero(df, item, 1, rng)[0]
    log.note(
        f"Life Event (J1) Causes Sense of Horror = 1 => a Life Events item > 0: {int(rows.sum())} rows changed"
    )

    recent, lifetime = "Number of Psych Admits (recent)", "Number of Psych Admits (lifetime)"
    rows = df[recent] > df[lifetime]
    df.loc[rows, lifetime] = df.loc[rows, recent]
    log.note(f"{recent} <= {lifetime}: {int(rows.sum())} rows changed {lifetime}")

    log.note(
        "Skipped: Age at First Hospitalization <= age at admission. Its codes 1-5 are age "
        "bands whose limits are not in the schema, so the rule cannot be checked."
    )
    log.note(
        "Holds by construction: positive screens <= screens run (screens = max over the "
        "panel of neg + pos >= each drug's pos)."
    )


def lab_label(
    values: pd.Series, count: pd.Series, gender: pd.Series, spec
) -> tuple[pd.Series, str]:
    """missing/low/normal/high for one analyte, and a description of the range used."""
    if spec is None:
        low, high = values.quantile([0.025, 0.975]).to_numpy()
        lows, highs = pd.Series(low, index=values.index), pd.Series(high, index=values.index)
        used = f"observed central 95%: {low:.3g}-{high:.3g}"
    elif isinstance(spec, dict):
        union = (min(r[0] for r in spec.values()), max(r[1] for r in spec.values()))
        lows = gender.map({g: r[0] for g, r in spec.items()}).fillna(union[0])
        highs = gender.map({g: r[1] for g, r in spec.items()}).fillna(union[1])
        used = (
            ", ".join(f"{g} {r[0]:g}-{r[1]:g}" for g, r in spec.items())
            + f", other {union[0]:g}-{union[1]:g}"
        )
    else:
        lows = pd.Series(spec[0], index=values.index)
        highs = pd.Series(spec[1], index=values.index)
        used = f"{spec[0]:g}-{spec[1]:g}"
    if ((count > 0) != values.notna()).any():
        raise ValueError(f"{values.name}: count and value disagree on which rows were measured")
    label = pd.Series("normal", index=values.index, dtype=object)
    label[values < lows] = "low"
    label[values > highs] = "high"
    label[count == 0] = "missing"
    # A level this rare makes generators miss it and HPO trials fail the
    # missing-category screen, so it joins "normal".
    sizes = label.value_counts()
    rare = [lvl for lvl in ("low", "high") if 0 < sizes.get(lvl, 0) < MIN_LEVEL_ROWS]
    label[label.isin(rare)] = "normal"
    if rare:
        used += f"; {'/'.join(rare)} under {MIN_LEVEL_ROWS} rows merged into normal"
    return label, used


def consolidate(df: pd.DataFrame, log: ChangeLog) -> tuple[pd.DataFrame, dict[str, str]]:
    """Consolidate columns; return the new frame and the kind of each added column."""
    added: dict[str, str] = {}
    dropped: list[str] = []
    log.section("B. Consolidation")

    # Labs.
    # nb_visit_since_last is a visit count, not a lab, and stays numeric.
    analytes = sorted([*LAB_RANGES, *DROPPED_LABS])
    log.note(
        "lab values are not normalized: medians match clinical SI units except HDL (about 9x); "
        "several sit outside the healthy range, as the level counts below show"
    )
    for panel, members in LAB_PANELS.items():
        counts = df[[f"{m}_count" for m in members]]
        measured = counts > 0
        log.note(
            f"{panel} = max of {len(members)} counts; members agree on measured/not in "
            f"{(measured.nunique(axis=1) == 1).mean():.1%} of rows, on the exact count in "
            f"{(counts.nunique(axis=1) == 1).mean():.1%}"
        )
        df[panel] = counts.max(axis=1)
        added[panel] = "continuous"
    for analyte in analytes:
        dropped += [f"{analyte}_mean", f"{analyte}_first"]
        if analyte in DROPPED_LABS:
            dropped.append(f"{analyte}_last")
            continue
        label, used = lab_label(
            df[f"{analyte}_last"], df[f"{analyte}_count"], df["gender"], LAB_RANGES[analyte]
        )
        df[f"{analyte}_last"] = label
        counts = label.value_counts().reindex(LAB_LEVELS, fill_value=0)
        log.note(f"{analyte}_last -> {'/'.join(LAB_LEVELS)} ({used}): {counts.to_dict()}")
    for members in LAB_PANELS.values():
        dropped += [f"{m}_count" for m in members]

    # Drug screens.
    totals = {d: df[f"{d}_neg"] + df[f"{d}_pos"] for d in UDS_DRUGS + AP_DRUGS + ["cannab"]}
    for name, drugs in SCREEN_COUNTS.items():
        df[name] = pd.concat([totals[d] for d in drugs], axis=1).max(axis=1)
        added[name] = "continuous"
    for name, drugs in POSITIVE_GROUPS.items():
        df[name] = (df[[f"{d}_pos" for d in drugs]] > 0).any(axis=1).astype(int)
        added[name] = "categorical"
    dropped += [c for c in df.columns if c.endswith(("_neg", "_pos")) and c not in added]
    log.note(
        f"drug screens: {', '.join(SCREEN_COUNTS)} (screens run) and {len(POSITIVE_GROUPS)} 0/1 "
        f"positive groups replace {2 * len(totals)} per-drug neg/pos counts"
    )

    # Medication classes.
    all_classes = MED_CORE + [c for group in MED_GROUPS.values() for c in group]
    df["n_med_classes"] = (df[all_classes] > 0).sum(axis=1)
    added["n_med_classes"] = "continuous"
    for c in MED_CORE:
        df[c] = (df[c] > 0).astype(int)
    for name, classes in MED_GROUPS.items():
        df[name] = (df[classes] > 0).any(axis=1).astype(int)
        added[name] = "categorical"
        dropped += classes
    log.note(
        f"medication classes: {len(MED_CORE)} kept as 0/1, {len(all_classes) - len(MED_CORE)} "
        f"grouped into {len(MED_GROUPS)} 0/1 flags, n_med_classes = classes present (of {len(all_classes)})"
    )

    # Duplicate sources.
    for kept, dup in DUPLICATE_SOURCES:
        table = pd.crosstab(df[kept], df[dup])
        share = table.max(axis=0).sum() / table.to_numpy().sum()
        log.note(
            f"{dup} vs {kept}: {share:.1%} of rows match the most common pairing; dropped {dup}"
        )
        dropped.append(dup)

    # Birthplace.
    born = df["Born in Canada"]
    continent = df["Continent of Birth"].replace({"North America": "North America (not Canada)"})
    location = pd.Series(np.nan, index=df.index, dtype=object)
    location[born == False] = "Outside Canada (continent unknown)"  # noqa: E712
    location[continent.notna()] = continent[continent.notna()]
    location[born == True] = "Canada"  # noqa: E712
    df["Location of Birth"] = location
    added["Location of Birth"] = "categorical"
    log.note(
        "Location of Birth from Born in Canada and Continent of Birth (no United States level: "
        f"the continent does not separate it): {location.value_counts(dropna=False).to_dict()}"
    )

    # Income sources: the first that applies, in this order.
    flags = df[[c for cols in INCOME_FLAGS.values() for c in cols] + [NO_INCOME]]
    source = pd.Series(np.nan, index=df.index, dtype=object)
    source[flags.notna().any(axis=1)] = "none"
    for level, cols in reversed(INCOME_FLAGS.items()):
        source[(df[cols] == 1).any(axis=1)] = level
    conflict = (df[NO_INCOME] == 1) & (source != "none")
    df["main_income_source"] = source
    added["main_income_source"] = "categorical"
    dropped += list(flags.columns)
    log.note(
        f"main_income_source (employment > government > other > none): "
        f"{source.value_counts(dropna=False).to_dict()}; {int(conflict.sum())} rows had "
        f"'{NO_INCOME}' = 1 next to an income source and take that source"
    )

    dropped += DROPPED_DEMOGRAPHICS
    df = df.drop(columns=dropped)

    # Rare levels.
    for col in RARE_LEVEL_COLUMNS:
        counts = df[col].value_counts()
        rare = counts[counts < MIN_LEVEL_ROWS]
        df.loc[df[col].isin(rare.index), col] = "Other"
        log.note(
            f"{col}: {len(rare)} levels under {MIN_LEVEL_ROWS} rows -> Other ({int(rare.sum())} rows)"
        )

    # Code 8 = not applicable.
    for col in NOT_APPLICABLE_8:
        rows = df[col] == 8
        df.loc[rows, col] = np.nan
        log.note(f"{col}: code 8 (not applicable) -> missing, {int(rows.sum())} rows")

    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant)
    log.note(f"constant columns dropped: {constant or 'none'}")

    log.section("Columns")
    log.note(f"dropped ({len(set(dropped + constant))}): {sorted(set(dropped + constant))}")
    log.note(f"added ({len(added)}): {sorted(added)}")
    return df, added


def build_schema(
    df: pd.DataFrame, old: pd.DataFrame, added: dict[str, str], log: ChangeLog
) -> pd.DataFrame:
    """Schema for the 1.3 columns: kind and ordinal order."""
    kinds = dict(zip(old["column"], old["kind"], strict=True))
    orders = dict(zip(old["column"], old["ordinal_order"], strict=True))
    lab_columns = [f"{analyte}_last" for analyte in LAB_RANGES]
    rows = []
    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        if col in added:
            kind, order = added[col], None
        elif col in MED_CORE or col in lab_columns:
            kind, order = "categorical", None
        else:
            kind = kinds[col]
            order = orders[col] if isinstance(orders[col], str) else None
        if col in NEW_ORDINAL_ORDERS:
            order = json.dumps(NEW_ORDINAL_ORDERS[col])
        rows.append({"column": col, "kind": kind, "ordinal_order": order})
    schema = pd.DataFrame(rows)
    retyped = [c for c in MED_CORE + lab_columns if kinds.get(c) != "categorical"]
    log.section("C. Schema")
    log.note(f"retyped continuous -> categorical: {retyped}")
    log.note(f"ordinal orders added: {sorted(NEW_ORDINAL_ORDERS)}")
    log.note(
        "labs are nominal (missing/low/normal/high): an ordinal order must list every "
        "observed level, so 'missing' cannot sit outside low < normal < high"
    )
    log.note(
        "left nominal: Available Social Supports items ('Not needed' has no place on the "
        "no < occasional < regular scale) and Education (codes 1-6, 8, 9 with 7 absent)"
    )
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/simulated"))
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir / "sim_v1.2.csv", low_memory=False)
    old_schema = pd.read_csv(args.data_dir / "sim_v1.2_variable_schema.csv")
    log = ChangeLog()
    log.note(
        f"sim 1.2: {df.shape[0]} rows, {df.shape[1]} columns ({len(old_schema)} in the schema)"
    )

    rng = np.random.default_rng(SEED)
    enforce_dependencies(df, rng, log)
    df, added = consolidate(df, log)
    schema = build_schema(df, old_schema, added, log)

    log.section("Result")
    log.note(f"sim 1.3: {df.shape[0]} rows, {df.shape[1]} columns ({len(schema)} in the schema)")
    categorical = schema.loc[schema["kind"] == "categorical", "column"]
    rare = {
        c: sizes[sizes < MIN_LEVEL_ROWS].to_dict()
        for c in categorical
        if (sizes := df[c].value_counts()).lt(MIN_LEVEL_ROWS).any()
    }
    log.note(
        f"categorical levels still under {MIN_LEVEL_ROWS} rows (left as they are): {rare or 'none'}"
    )
    df.to_csv(args.data_dir / "sim_v1.3.csv", index=False)
    schema.to_csv(args.data_dir / "sim_v1.3_variable_schema.csv", index=False)
    log.print()


if __name__ == "__main__":
    main()
