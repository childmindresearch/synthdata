"""Interactive guide to multiclass classification metrics on an imbalanced target.

Run it with ``uv run marimo edit notebooks/classification_metrics.py``
(or ``marimo run`` for a read-only app).
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.utils.class_weight import compute_sample_weight

    return (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        compute_sample_weight,
        f1_score,
        mo,
        np,
        pd,
        plt,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Which metric should judge a readmission classifier?

    This notebook compares five ways of scoring a multiclass classifier on an **imbalanced
    clinical target**, using a readmission example with three classes:

    | Class | Meaning | Share of patients | Patients in our test set of 1,000 |
    |---|---|---|---|
    | `none` | not readmitted | 82% | 820 |
    | `late` | readmitted after 30 days | 13% | 130 |
    | `early` | readmitted within 30 days | 5% | 50 |

    The 82 / 13 / 5 split is a generic example of how lopsided clinical outcomes usually are:
    the class we care most about (`early`) is the rarest.

    The metrics:

    - **macro-F1**: the F1 score of each class, averaged with equal weight.
    - **balanced accuracy**: the recall (sensitivity) of each class, averaged with equal weight.
    - **Cohen's kappa**: agreement between predictions and truth beyond what chance would give,
      shown two ways: the default (every patient counts once) and with
      **inverse-frequency class weights** (each class counts as much as the others in total).
    - **macro AUROC**: area under the ROC curve of each class against the rest, averaged.
    - **macro AUPRC**: area under the precision-recall curve (average precision) of each class
      against the rest, averaged.

    Plain **accuracy** is shown too, as a warning: it is the metric that looks best when a model
    does nothing useful.

    The notebook goes from definitions, to fixed scenarios with exact numbers, to two
    experiments with sliders, and ends with a playground where you type in your own confusion
    matrix.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Two families of metrics

    A classifier gives every patient a **score** per class (for example
    `none 0.70, late 0.20, early 0.10`). To get a **label** we apply a decision rule, usually
    "pick the class with the highest score" (the *argmax*). The two families of metrics look at
    different things:

    | | Threshold-dependent (label) metrics | Ranking (score) metrics |
    |---|---|---|
    | Metrics here | macro-F1, balanced accuracy, Cohen's kappa, accuracy | macro AUROC, macro AUPRC |
    | Input | predicted labels only, so one confusion matrix | the scores, before any decision rule |
    | Question answered | "How good are the decisions this rule makes?" | "Does the model rank patients of a class above the others?" |
    | Changes when you move the threshold | yes, often a lot | no |
    | Strength | matches how the model would be used once a rule is fixed; easy to explain | judges the model, not one rule; stable; no threshold to tune |
    | Weakness | depends on a threshold that may be arbitrary; small threshold changes can flip rankings between models | says nothing about which threshold to use; a model can rank well and still make poor decisions at argmax |

    ### Threshold-dependent metrics, from the confusion matrix

    Write $C_{ij}$ for the number of patients of true class $i$ predicted as class $j$, $N$
    for all patients and $K = 3$ for the number of classes. For each class $k$:

    - $\text{TP}_k = C_{kk}$ (true class $k$, predicted $k$)
    - $\text{FN}_k = \sum_{j \ne k} C_{kj}$ (true class $k$, predicted something else; row $k$ minus the diagonal)
    - $\text{FP}_k = \sum_{i \ne k} C_{ik}$ (predicted $k$, true class something else; column $k$ minus the diagonal)
    - recall $R_k = \text{TP}_k / (\text{TP}_k + \text{FN}_k)$, precision $P_k = \text{TP}_k / (\text{TP}_k + \text{FP}_k)$,
      $F1_k = 2 P_k R_k / (P_k + R_k)$ (taken as 0 when the class is never predicted)

    Then

    $$\text{macro-F1} = \frac{1}{K}\sum_k F1_k \qquad \text{balanced accuracy} = \frac{1}{K}\sum_k R_k \qquad \text{accuracy} = \frac{\sum_k C_{kk}}{N}$$

    **Cohen's kappa** compares the observed agreement $p_o$ (accuracy) with the agreement
    $p_e$ that two independent raters with the same label frequencies would reach by chance:

    $$\kappa = \frac{p_o - p_e}{1 - p_e}, \qquad p_e = \sum_k \frac{\text{row}_k}{N}\cdot\frac{\text{col}_k}{N}$$

    **Kappa with inverse-frequency class weights** gives each patient of class $k$ the weight
    $N / (K \cdot n_k)$ (scikit-learn's `compute_sample_weight("balanced")`), so every class
    adds up to the same total weight $N/K$. Then $p_o$ becomes the mean recall, which is the
    balanced accuracy, and $p_e$ becomes exactly $1/K$ whatever the model predicts. So

    $$\kappa_{\text{weighted}} = \frac{\text{balanced accuracy} - 1/K}{1 - 1/K}$$

    It is a rescaled balanced accuracy, with chance moved to 0. It adds no new information
    beyond balanced accuracy, but it is easier to read: 0 is chance, 1 is perfect, below 0 is
    worse than chance. The default kappa is different: its chance term depends on how often
    the model predicts each class, so it punishes a model that over-predicts a rare class.

    ### Ranking metrics, from the scores

    For each class $k$, treat "class $k$" as positive and the other classes as negative, sort
    patients by their score for $k$, and sweep a threshold from high to low.

    - **AUROC** is the area under recall vs false-positive rate. It equals the probability that
      a random positive patient gets a higher score than a random negative one. Chance is 0.5
      for every class, whatever its share.
    - **AUPRC** (average precision) is the area under precision vs recall. Chance equals the
      class's share: 0.82 for `none`, 0.13 for `late`, 0.05 for `early`. Averaged over the
      three classes, chance is $(0.82 + 0.13 + 0.05)/3 = 1/3$.

    Both are averaged over classes with equal weight ("macro", one-vs-rest), as in the
    pipeline's TSTR scores (`synthdata/evaluation/tstr.py`).

    | Metric | Chance level here | Weakness to keep in mind |
    |---|---|---|
    | accuracy | 0.82 (always say `none`) | rewards ignoring the minority classes |
    | macro-F1 | about 0.30 (always say `none`) | needs a threshold; a class never predicted scores 0, so it is jumpy |
    | balanced accuracy | 0.33 | rewards a model that flags everyone as a rare class (recall is high, precision is ignored) |
    | kappa (default) | 0 | depends on the predicted class frequencies; can be low for a useful model with a different label mix |
    | kappa (class weights) | 0 | only a rescaled balanced accuracy, with the same blind spot for false positives |
    | macro AUROC | 0.5 | looks good on rare classes even when most flagged patients are false alarms |
    | macro AUPRC | 0.33 | depends on class shares, so it cannot be compared across datasets with different shares |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Getting scores from a confusion matrix

    A confusion matrix holds only labels, so it is enough for the threshold metrics but not
    for AUROC or AUPRC, which need scores. To show all five metrics side by side, this notebook
    **simulates scores** that agree with each confusion matrix:

    1. For each patient, draw one raw score per class from a normal distribution, and add a
       bonus called the **separation** to the score of the patient's true class.
       Separation 0 means the scores carry no information beyond the label; larger values
       mean patients of a class get clearly higher scores for that class.
    2. Raise the score of the class the confusion matrix says was **predicted** just above
       the others, so that the argmax reproduces the confusion matrix exactly.
    3. Turn the raw scores into probabilities with a softmax.

    Because of step 2 the threshold metrics are computed from exactly the matrix you see.
    The ranking metrics depend on both the matrix and the separation, which is the point:
    **one confusion matrix is compatible with many different AUROC and AUPRC values.** The
    simulation uses a fixed random seed, so the numbers are reproducible; they would move a
    little with another seed.
    """)
    return


@app.cell
def _(
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    compute_sample_weight,
    f1_score,
    np,
    roc_auc_score,
):
    CLASSES = ["none", "late", "early"]
    K = len(CLASSES)
    SEED = 0

    def expand(matrix):
        """Turn a confusion matrix (rows true, columns predicted) into label arrays."""
        m = np.asarray(matrix, dtype=int)
        y_true, y_pred = [], []
        for i in range(K):
            for j in range(K):
                y_true += [i] * m[i, j]
                y_pred += [j] * m[i, j]
        return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)

    def simulate_scores(y_true, y_pred, separation, seed=SEED):
        """Class probabilities whose argmax equals ``y_pred`` (see section 2)."""
        rng = np.random.default_rng(seed)
        n = len(y_true)
        z = rng.normal(size=(n, K))
        z[np.arange(n), y_true] += separation
        rows = np.arange(n)
        others = z.copy()
        others[rows, y_pred] = -np.inf
        need = z[rows, y_pred] <= others.max(axis=1)
        z[rows[need], y_pred[need]] = others.max(axis=1)[need] + rng.uniform(0.01, 0.3, need.sum())
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def macro_auprc(y_true, proba):
        """Mean one-vs-rest average precision over the classes present (as the pipeline)."""
        scored = [k for k in range(K) if (y_true == k).any()]
        return float(np.mean([average_precision_score(y_true == k, proba[:, k]) for k in scored]))

    def macro_auroc(y_true, proba):
        scored = [k for k in range(K) if 0 < (y_true == k).sum() < len(y_true)]
        if not scored:
            return float("nan")
        return float(np.mean([roc_auc_score(y_true == k, proba[:, k]) for k in scored]))

    def kappa(y_true, y_pred, weighted):
        if len(np.unique(np.r_[y_true, y_pred])) < 2:
            return (
                0.0  # truth and predictions are the same single class: no agreement beyond chance
            )
        w = compute_sample_weight("balanced", y_true) if weighted else None
        return float(cohen_kappa_score(y_true, y_pred, labels=range(K), sample_weight=w))

    def threshold_metrics(y_true, y_pred):
        present = [k for k in range(K) if (y_true == k).any()]
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro-F1": float(
                f1_score(y_true, y_pred, labels=range(K), average="macro", zero_division=0)
            ),
            "balanced accuracy": float(
                np.mean([(y_pred[y_true == k] == k).mean() for k in present])
            )
            if present
            else float("nan"),
            "kappa (default)": kappa(y_true, y_pred, weighted=False),
            "kappa (class weights)": kappa(y_true, y_pred, weighted=True),
        }

    def ranking_metrics(y_true, proba):
        return {
            "macro AUROC": macro_auroc(y_true, proba),
            "macro AUPRC": macro_auprc(y_true, proba),
        }

    def all_metrics(matrix, separation, seed=SEED):
        y_true, y_pred = expand(matrix)
        proba = simulate_scores(y_true, y_pred, separation, seed)
        return {**threshold_metrics(y_true, y_pred), **ranking_metrics(y_true, proba)}

    # Sanity checks: the simulated scores reproduce the matrix, and the balanced-accuracy
    # helper agrees with scikit-learn.
    _yt, _yp = expand([[700, 90, 30], [30, 85, 15], [8, 10, 32]])
    assert (simulate_scores(_yt, _yp, 1.0).argmax(axis=1) == _yp).all()
    assert np.isclose(
        threshold_metrics(_yt, _yp)["balanced accuracy"], balanced_accuracy_score(_yt, _yp)
    )
    return (
        CLASSES,
        K,
        all_metrics,
        expand,
        ranking_metrics,
        simulate_scores,
        threshold_metrics,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Scenarios

    Each scenario is a confusion matrix for the same 1,000 test patients (rows are the true
    class, columns the predicted class, in the order `none`, `late`, `early`), plus a
    separation for the simulated scores.
    """)
    return


@app.cell
def _():
    SCENARIOS = {
        "A. Always predicts none": {
            "matrix": [[820, 0, 0], [130, 0, 0], [50, 0, 0]],
            "separation": 0.0,
            "story": "Predicts the majority class for everyone and its scores carry no information.",
        },
        "B. Always predicts none, informative scores": {
            "matrix": [[820, 0, 0], [130, 0, 0], [50, 0, 0]],
            "separation": 2.0,
            "story": "Same labels as A, but the scores do rank readmitted patients higher. A "
            "different threshold would find them.",
        },
        "C. Random guess at the class shares": {
            "matrix": [[672, 107, 41], [107, 17, 6], [41, 6, 3]],
            "separation": 0.0,
            "story": "Labels drawn at random with the 82/13/5 shares; no skill at all.",
        },
        "D. Random guess, equal shares": {
            "matrix": [[273, 274, 273], [43, 44, 43], [17, 16, 17]],
            "separation": 0.0,
            "story": "Labels drawn at random with equal shares; no skill, but many minority "
            "predictions.",
        },
        "E. Cautious, high precision": {
            "matrix": [[815, 4, 1], [95, 33, 2], [35, 3, 12]],
            "separation": 1.0,
            "story": "Flags few patients, and is usually right when it does, but misses most "
            "readmissions.",
        },
        "F. Alarm-happy, high sensitivity": {
            "matrix": [[480, 220, 120], [12, 105, 13], [2, 5, 43]],
            "separation": 1.0,
            "story": "Catches most readmissions, at the cost of flagging 340 of the 820 "
            "patients who were never readmitted.",
        },
        "G. Balanced, decent model": {
            "matrix": [[700, 90, 30], [30, 85, 15], [8, 10, 32]],
            "separation": 1.0,
            "story": "A reasonable trade-off between missed readmissions and false alarms.",
        },
        "H. Misses early, calls it late": {
            "matrix": [[790, 25, 5], [25, 100, 5], [5, 40, 5]],
            "separation": 1.0,
            "story": "Good on none and late, but labels most early readmissions as late: a "
            "near miss on an ordered outcome.",
        },
        "I. Near perfect": {
            "matrix": [[810, 8, 2], [6, 120, 4], [1, 4, 45]],
            "separation": 3.0,
            "story": "Almost every patient correctly labelled, with confident scores.",
        },
    }
    return (SCENARIOS,)


@app.cell
def _(CLASSES, SCENARIOS, all_metrics, mo, pd):
    def matrix_frame(matrix):
        """A confusion matrix as a small HTML table (rows true, columns predicted)."""
        df = pd.DataFrame(
            matrix,
            index=[f"true {c}" for c in CLASSES],
            columns=[f"pred {c}" for c in CLASSES],
        )
        return mo.Html(df.to_html(border=0))

    results = pd.DataFrame(
        {name: all_metrics(s["matrix"], s["separation"]) for name, s in SCENARIOS.items()}
    ).T
    results.index.name = "scenario"

    mo.vstack(
        [
            mo.md("### The confusion matrices"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(f"**{name}** (separation {s['separation']:g})"),
                            matrix_frame(s["matrix"]),
                            mo.md(f"<small>{s['story']}</small>"),
                        ]
                    )
                    for name, s in SCENARIOS.items()
                ],
                wrap=True,
                gap=1.5,
            ),
        ]
    )
    return matrix_frame, results


@app.cell
def _(mo, np, plt, results):
    def heatmap(df):
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.5))
        ax.imshow(df.to_numpy(dtype=float), cmap="coolwarm", vmin=-0.2, vmax=1, aspect="auto")
        ax.set_xticks(range(df.shape[1]), df.columns, rotation=30, ha="right")
        ax.set_yticks(range(df.shape[0]), df.index)
        for (i, j), v in np.ndenumerate(df.to_numpy(dtype=float)):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v < 0.05 or v > 0.85 else "black",
            )
        ax.axvline(4.5, color="black", lw=2)
        ax.set_title("Threshold metrics (left of the line) vs ranking metrics (right)")
        fig.tight_layout()
        return fig

    mo.vstack(
        [
            mo.md("### Every metric in every scenario"),
            mo.ui.table(results.round(3), selection=None, pagination=False),
            heatmap(results),
        ]
    )
    return


@app.cell
def _(
    CLASSES,
    average_precision_score,
    expand,
    np,
    plt,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    simulate_scores,
):
    CLASS_COLORS = ["#3b6fb6", "#e08a2c", "#c23b3b"]

    def curves(matrix, separation, title=""):
        """Per-class ROC and precision-recall curves for the simulated scores of a matrix."""
        y_true, y_pred = expand(matrix)
        proba = simulate_scores(y_true, y_pred, separation)
        fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(10, 4.2))
        roc_ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1, label="chance (AUROC 0.5)")
        aucs, aps = [], []
        for k, (name, color) in enumerate(zip(CLASSES, CLASS_COLORS, strict=True)):
            positive = y_true == k
            if not 0 < positive.sum() < len(y_true):
                continue
            fpr, tpr, _ = roc_curve(positive, proba[:, k])
            auc = roc_auc_score(positive, proba[:, k])
            roc_ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}: AUROC {auc:.2f}")
            precision, recall, _ = precision_recall_curve(positive, proba[:, k])
            ap = average_precision_score(positive, proba[:, k])
            pr_ax.step(
                recall, precision, where="post", color=color, lw=2, label=f"{name}: AUPRC {ap:.2f}"
            )
            share = positive.mean()
            pr_ax.axhline(share, color=color, ls=":", lw=1)
            pr_ax.text(1.0, share, f" chance {share:.2f}", color=color, va="center", fontsize=8)
            aucs.append(auc)
            aps.append(ap)
        roc_ax.set(
            xlabel="False-positive rate (share of other patients flagged)",
            ylabel="Recall (share of this class found)",
            title=f"ROC, one class vs the rest (macro AUROC {np.mean(aucs):.2f})",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        pr_ax.set(
            xlabel="Recall (share of this class found)",
            ylabel="Precision (share of flagged who are this class)",
            title=f"Precision-recall (macro AUPRC {np.mean(aps):.2f})",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        roc_ax.legend(loc="lower right", fontsize=8)
        pr_ax.legend(loc="upper right", fontsize=8)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return fig

    return (curves,)


@app.cell
def _(SCENARIOS, mo):
    curve_pick = mo.ui.dropdown(
        options=list(SCENARIOS), value="G. Balanced, decent model", label="Scenario"
    )
    mo.vstack(
        [
            mo.md(
                r"""
    ### ROC and precision-recall curves

    AUROC and AUPRC are areas under curves. Each curve comes from sorting the patients by
    their score for one class and lowering the threshold step by step: every step flags more
    patients, finding more of the class (recall goes up) but also flagging more patients of
    the other classes.

    - **ROC curve (left):** recall against the false-positive rate. The dotted diagonal is a
      model that ranks at random (area 0.5). A curve hugging the top-left corner finds most of
      the class before flagging many others.
    - **Precision-recall curve (right):** precision against recall. The dotted line in each
      colour is that class's share, the precision of random flagging. For `early` it sits at
      0.05, so the curve has a long way to climb.

    The false-positive rate divides by all patients of the other classes (950 for `early`), so
    100 false alarms move the ROC curve only about 0.1 along its x-axis, while the same 100
    false alarms next to all 50 true `early` patients drop precision to a third. Compare the `early` curve
    in both panels to see why AUROC looks more flattering than AUPRC on a rare class. Try
    A and B (same labels, different scores), and F against G.
    """
            ),
            curve_pick,
        ]
    )
    return (curve_pick,)


@app.cell
def _(SCENARIOS, curve_pick, curves):
    _s = SCENARIOS[curve_pick.value]
    curves(_s["matrix"], _s["separation"], f"{curve_pick.value} (separation {_s['separation']:g})")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What the scenarios show

    The threshold metrics above are exact (they come straight from the matrices). The AUROC
    and AUPRC values come from the simulated scores, so read them as illustrations.

    - **Accuracy is misleading.** A (always `none`) scores 0.82, above the alarm-happy F
      (0.63) and level with the balanced G (0.82), although A never finds a readmission. The
      cautious E (0.86) beats G on accuracy while missing three quarters of the early readmissions.
    - **Macro-F1 is not 0 for a useless model.** A gets 0.30: the `none` class alone has F1
      0.90, and that third of the average does not go away. Random guessing at the class shares
      (C) gets 0.34, at equal shares (D) 0.25. Its chance level depends on the class shares and
      on the guessing strategy, which makes a macro-F1 value hard to read on its own.
    - **Balanced accuracy and class-weighted kappa reward over-flagging.** F flags 506 patients
      as readmitted, 340 of whom were never readmitted, yet it has the best balanced accuracy of the model
      scenarios (0.75, above G's 0.72), because recall is all these two metrics look at.
      Class-weighted kappa is exactly $(\text{BA} - 1/3)/(2/3)$ in every row, so it always
      ranks scenarios the same way as balanced accuracy.
    - **Default kappa and macro-F1 punish over-flagging.** F's default kappa (0.32) is well
      below G's (0.50), because predicting many minority labels raises the chance agreement
      $p_e$ it is compared with, and the false alarms drag down precision and so F1. Default
      kappa also stays at 0 for all three no-skill scenarios (A, C, D), which macro-F1 does not.
    - **The two kappas can disagree in both directions.** H, which gets `none` and `late` right
      but calls most early readmissions late, has the highest default kappa of the model
      scenarios (0.66), because default kappa is dominated by the large classes. With class
      weights it drops to 0.42, because the `early` class, with recall 0.10, now counts as much
      as the others. F goes the other way (0.32 default, 0.63 weighted).
    - **A and B have identical threshold metrics but very different ranking metrics.** B's
      scores put readmitted patients higher (AUROC 0.91, AUPRC 0.69, against 0.53 and 0.34 for
      A), so the model has signal; the argmax rule just never uses it. This is why ranking
      metrics are the better way to compare **models**, and threshold metrics the better way to
      judge a **decision rule**.
    - **Ranking metrics barely separate F and G** (AUROC 0.87 vs 0.88, AUPRC 0.67 vs 0.67) while
      the threshold metrics disagree about them. Both models can rank patients about as well;
      what differs is mostly where the decision threshold sits (section 5 makes this concrete).
    - **AUROC is generous on rare classes.** In G, the `early` class has AUROC 0.90 but AUPRC
      only 0.51. AUROC asks whether early patients rank above the other 950, which they mostly
      do; AUPRC asks how many of the patients ranked high are really early, and with 950
      negatives even a small false-positive rate means many false alarms. On a 5% class AUPRC
      is the more honest of the two. Its chance level is that class's share (0.05), so a value
      of 0.51 is still about ten times chance.
    - **H shows a blind spot of every metric here.** Calling an early readmission "late" counts
      exactly as wrong as calling it "none". If the classes are ordered, quadratic-weighted
      kappa (which the pipeline uses when the target is declared ordinal) gives partial credit
      for near misses.
    """)
    return


@app.cell
def _(SCENARIOS, mo):
    worked_pick = mo.ui.dropdown(
        options=list(SCENARIOS), value="F. Alarm-happy, high sensitivity", label="Scenario"
    )
    mo.vstack(
        [
            mo.md(
                "## 4. A worked calculation\n\nPick a scenario to see every threshold "
                "metric computed by hand from its confusion matrix."
            ),
            worked_pick,
        ]
    )
    return (worked_pick,)


@app.cell
def _(CLASSES, K, SCENARIOS, matrix_frame, mo, np, pd, worked_pick):
    def worked(matrix):
        m = np.asarray(matrix, dtype=float)
        n = m.sum()
        rows, cols, tp = m.sum(axis=1), m.sum(axis=0), np.diag(m)
        fn, fp = rows - tp, cols - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            recall = np.where(rows > 0, tp / rows, np.nan)
            precision = np.where(cols > 0, tp / cols, 0.0)
            f1 = np.where(
                precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
            )
        per_class = pd.DataFrame(
            {
                "patients (row sum)": rows.astype(int),
                "predicted (column sum)": cols.astype(int),
                "TP": tp.astype(int),
                "FN": fn.astype(int),
                "FP": fp.astype(int),
                "recall": recall.round(3),
                "precision": precision.round(3),
                "F1": f1.round(3),
            },
            index=CLASSES,
        )
        p_o = tp.sum() / n
        p_e = float((rows / n) @ (cols / n))
        present = rows > 0
        ba = float(np.nanmean(recall[present]))
        w = np.where(rows > 0, n / (present.sum() * np.where(rows > 0, rows, 1)), 0.0)
        mw = m * w[:, None]
        nw = mw.sum()
        p_o_w = np.trace(mw) / nw
        p_e_w = float((mw.sum(axis=1) / nw) @ (mw.sum(axis=0) / nw))
        terms = " + ".join(f"{r / n:.3f}·{c / n:.3f}" for r, c in zip(rows, cols, strict=True))
        text = rf"""
    **Macro-F1** = mean of the F1 column = ({" + ".join(f"{v:.3f}" for v in f1)}) / {K} = **{f1.mean():.3f}**

    **Balanced accuracy** = mean of the recall column = **{ba:.3f}**

    **Default kappa.** $p_o$ = accuracy = {int(tp.sum())} / {int(n)} = {p_o:.3f}.
    $p_e$ = Σ (row share × column share) = {terms} = {p_e:.3f}.
    κ = ({p_o:.3f} − {p_e:.3f}) / (1 − {p_e:.3f}) = **{(p_o - p_e) / (1 - p_e):.3f}**

    **Kappa with class weights.** Each row is scaled so that it sums to N/K
    (weights {", ".join(f"{v:.2f}" for v in w)} per patient of {", ".join(CLASSES)}).
    $p_o$ = {p_o_w:.3f} (the balanced accuracy), $p_e$ = {p_e_w:.3f} (= 1/K).
    κ = ({p_o_w:.3f} − {p_e_w:.3f}) / (1 − {p_e_w:.3f}) = **{(p_o_w - p_e_w) / (1 - p_e_w):.3f}**
    """
        return per_class, text

    _per_class, _text = worked(SCENARIOS[worked_pick.value]["matrix"])
    mo.vstack(
        [
            matrix_frame(SCENARIOS[worked_pick.value]["matrix"]),
            mo.ui.table(_per_class, selection=None, pagination=False),
            mo.md(_text),
        ]
    )
    return


@app.cell
def _(mo):
    shift = mo.ui.slider(
        start=-2.0,
        stop=3.0,
        step=0.25,
        value=0.0,
        show_value=True,
        label="Boost added to the late and early raw scores before the argmax",
    )
    mo.vstack(
        [
            mo.md(
                r"""
    ## 5. Same model, different threshold

    Here the scores of scenario G stay fixed and only the decision rule moves: a boost is
    added to the `late` and `early` scores before taking the argmax. A positive boost flags
    more patients as readmitted (moving towards F), a negative one fewer (towards A).

    Watch the threshold metrics change while AUROC and AUPRC stay exactly the same. This is
    the reason the pipeline treats macro AUPRC as its main TSTR score: when two generators
    are compared through a classifier trained with class-balanced weights, an argmax metric
    such as macro-F1 can move because the classifier's implicit threshold moved, not because
    the synthetic data got better or worse (see the TSTR rows of `docs/verification.md`).
    """
            ),
            shift,
        ]
    )
    return (shift,)


@app.cell
def _(
    K,
    SCENARIOS,
    expand,
    matrix_frame,
    mo,
    np,
    pd,
    ranking_metrics,
    shift,
    simulate_scores,
    threshold_metrics,
):
    _g = SCENARIOS["G. Balanced, decent model"]
    _yt, _yp = expand(_g["matrix"])
    _proba = simulate_scores(_yt, _yp, _g["separation"])
    _logits = np.log(_proba) + np.array([0.0, shift.value, shift.value])
    _pred = _logits.argmax(axis=1)
    _m = pd.crosstab(
        pd.Categorical(_yt, categories=range(K)),
        pd.Categorical(_pred, categories=range(K)),
        dropna=False,
    ).to_numpy()
    _row = {**threshold_metrics(_yt, _pred), **ranking_metrics(_yt, _proba)}
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("**Confusion matrix at this threshold**"),
                    matrix_frame(_m),
                ]
            ),
            mo.vstack(
                [
                    mo.md("**Metrics**"),
                    mo.ui.table(
                        pd.DataFrame({"value": _row}).round(3).rename_axis("metric").reset_index(),
                        selection=None,
                        pagination=False,
                    ),
                ]
            ),
        ],
        widths=[1, 1],
    )
    return


@app.cell
def _(mo):
    sep = mo.ui.slider(
        start=0.0,
        stop=4.0,
        step=0.25,
        value=1.0,
        show_value=True,
        label="Separation of the simulated scores",
    )
    mo.vstack(
        [
            mo.md(
                r"""
    ## 6. Same confusion matrix, different scores

    Now the opposite experiment: the confusion matrix of scenario G stays fixed, and only the
    separation of the simulated scores changes. The threshold metrics cannot move, because the
    labels do not change. AUROC and AUPRC rise with the separation: within each cell of the
    matrix, patients of the right class are ranked higher.

    Together with section 5 this shows that the two families measure different things, and
    neither can be computed from the other.
    """
            ),
            sep,
        ]
    )
    return (sep,)


@app.cell
def _(SCENARIOS, all_metrics, curves, mo, pd, sep):
    _matrix = SCENARIOS["G. Balanced, decent model"]["matrix"]
    _row = all_metrics(_matrix, sep.value)
    mo.vstack(
        [
            mo.ui.table(
                pd.DataFrame({"value": _row}).round(3).rename_axis("metric").reset_index(),
                selection=None,
                pagination=False,
            ),
            curves(_matrix, sep.value),
        ]
    )
    return


@app.cell
def _(SCENARIOS, mo):
    preset = mo.ui.dropdown(
        options=list(SCENARIOS), value="G. Balanced, decent model", label="Start from"
    )
    mo.vstack(
        [
            mo.md(
                r"""
    ## 7. Playground: your own confusion matrix

    Pick a scenario to start from, then edit any cell (rows are the true class, columns the
    predicted class). The class shares follow the row sums, so keep them at 820 / 130 / 50 to
    stay with the 82 / 13 / 5 example, or change them to see how the metrics react to a
    different imbalance.
    """
            ),
            preset,
        ]
    )
    return (preset,)


@app.cell
def _(CLASSES, SCENARIOS, mo, preset):
    _start = SCENARIOS[preset.value]
    cells = mo.ui.array(
        [
            mo.ui.number(start=0, stop=100_000, step=1, value=_start["matrix"][i][j])
            for i in range(3)
            for j in range(3)
        ]
    )
    play_sep = mo.ui.slider(
        start=0.0,
        stop=4.0,
        step=0.25,
        value=_start["separation"],
        show_value=True,
        label="Separation of the simulated scores (ranking metrics only)",
    )
    _header = mo.hstack(
        [mo.md("**true ↓ / predicted →**")] + [mo.md(f"**{c}**") for c in CLASSES], widths="equal"
    )
    _rows = [
        mo.hstack(
            [mo.md(f"**{CLASSES[i]}**")] + [cells[3 * i + j] for j in range(3)], widths="equal"
        )
        for i in range(3)
    ]
    mo.vstack([_header, *_rows, play_sep])
    return cells, play_sep


@app.cell
def _(CLASSES, K, all_metrics, cells, curves, mo, np, pd, play_sep, results):
    _m = np.array(cells.value, dtype=int).reshape(K, K)
    _rows = _m.sum(axis=1)
    if _m.sum() == 0:
        _out = mo.callout(mo.md("The matrix is empty: enter some patients."), kind="warn")
    else:
        _row = all_metrics(_m.tolist(), play_sep.value)
        _shares = ", ".join(f"{c} {r / _m.sum():.0%}" for c, r in zip(CLASSES, _rows, strict=True))
        _table = pd.DataFrame(
            {
                "your matrix": _row,
                "always none (A)": results.loc["A. Always predicts none"],
                "near perfect (I)": results.loc["I. Near perfect"],
            }
        ).round(3)
        _notes = []
        if (_rows == 0).any():
            _notes.append(
                "A class with no patients is left out of the macro averages (as in the pipeline)."
            )
        _out = mo.vstack(
            [
                mo.md(f"{_m.sum()} patients; class shares {_shares}."),
                mo.ui.table(
                    _table.rename_axis("metric").reset_index(), selection=None, pagination=False
                ),
                *[mo.md(f"<small>{n}</small>") for n in _notes],
                curves(_m.tolist(), play_sep.value),
            ]
        )
    mo.output.replace(_out)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Takeaways

    - **Never judge an imbalanced classifier by accuracy.** Always show it next to a constant
      "majority class" baseline.
    - **Compare models with a ranking metric; on rare classes prefer AUPRC.** It does not depend
      on a threshold, and unlike AUROC it notices when the flagged group is full of false
      alarms. Report its chance level (the class share) next to it.
    - **Judge a decision rule with threshold metrics, and look at more than one.** Balanced
      accuracy (and kappa with class weights, which is the same number rescaled) shows how many
      of each class are found; default kappa and macro-F1 also punish false alarms. Disagreement
      between them is a hint that the model trades precision for recall.
    - **A threshold metric that moves while AUPRC does not** usually means the decision
      threshold moved, not the model's quality.
    - **Kappa's two forms answer different questions.** Default kappa: "how much better than
      chance are these labels, given how often each label is used?" Kappa with inverse-frequency
      weights: "how much better than chance is the average recall?"

    ### How this maps to the pipeline

    The holdout TSTR scores in `tstr_holdout.csv` are computed as here: macro-F1, balanced
    accuracy and Cohen's kappa on the argmax of a classifier fitted with class-balanced
    sample weights, and macro AUPRC as the mean one-vs-rest average precision over the classes
    present. Note the difference between the two uses of class weights: the pipeline weights
    the classifier's **training** rows, which shifts its argmax towards the rare classes, and
    then reports the **default** kappa (quadratic-weighted when the target is declared ordinal).
    The class-weighted kappa in this notebook weights the **test** patients instead.
    `docs/metrics-guide.md` describes every score the pipeline reports.
    """)
    return


if __name__ == "__main__":
    app.run()
