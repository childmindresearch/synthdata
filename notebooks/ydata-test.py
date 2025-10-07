import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Tabular Synthetic Data Generation with Gaussian Mixture and CTGAN
    - This notebook is an example of how to use a synthetic data generation methods based on [GMM](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) to generate synthetic tabular data with numeric and categorical features.
    - This notebook is an example of how to use CTGAN (implemented accordingly with the [paper](https://arxiv.org/pdf/1907.00503.pdf)) to generate synthetic tabular data with numeric and categorical features.
    """
    )
    return


@app.cell
def _():
    import os

    import altair as alt
    import marimo as mo
    import pandas as pd
    from pmlb import fetch_data
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer

    # Path to models directory
    MODELS_DIR = "../models"

    # Sample size for visualization
    SAMPLE_SIZE = 1000

    return (
        MODELS_DIR,
        SAMPLE_SIZE,
        ModelParameters,
        RegularSynthesizer,
        TrainParameters,
        alt,
        fetch_data,
        mo,
        os,
        pd,
    )


@app.cell
def _(pd):
    def sample_column(
        df: pd.DataFrame, col: str, sample_size: int, random_state: int = 42
    ) -> pd.Series:
        """Sample a column from a dataframe."""
        return (
            df[col]
            .sample(n=sample_size, random_state=random_state)
            .reset_index(drop=True)
        )

    def prepare_numeric_data(
        data: pd.DataFrame,
        gmm_data: pd.DataFrame,
        ctgan_data: pd.DataFrame,
        col: str,
        sample_size: int,
    ) -> pd.DataFrame:
        """Prepare combined data for numeric column visualization."""
        original_sample = sample_column(data, col, sample_size)
        gmm_sample = sample_column(gmm_data, col, sample_size)
        ctgan_sample = sample_column(ctgan_data, col, sample_size)

        original_df = pd.DataFrame({col: original_sample, "Model": "Original"})
        gmm_df = pd.DataFrame({col: gmm_sample, "Model": "GMM"})
        ctgan_df = pd.DataFrame({col: ctgan_sample, "Model": "CTGAN"})

        return pd.concat([original_df, gmm_df, ctgan_df])

    def prepare_categorical_data(
        data: pd.DataFrame,
        gmm_data: pd.DataFrame,
        ctgan_data: pd.DataFrame,
        col: str,
        sample_size: int,
    ) -> pd.DataFrame:
        """Prepare combined data for categorical column visualization."""
        original_sample = sample_column(data, col, sample_size)
        gmm_sample = sample_column(gmm_data, col, sample_size)
        ctgan_sample = sample_column(ctgan_data, col, sample_size)

        # Get value counts for each dataset
        original_counts = original_sample.value_counts().reset_index()
        original_counts.columns = [col, "count"]
        original_counts["Model"] = "Original"

        gmm_counts = gmm_sample.value_counts().reset_index()
        gmm_counts.columns = [col, "count"]
        gmm_counts["Model"] = "GMM"

        ctgan_counts = ctgan_sample.value_counts().reset_index()
        ctgan_counts.columns = [col, "count"]
        ctgan_counts["Model"] = "CTGAN"

        return pd.concat([original_counts, gmm_counts, ctgan_counts])

    return prepare_categorical_data, prepare_numeric_data, sample_column


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Dataset

    - The data used is the [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income) which we will fetch by importing the `pmlb` library (a wrapper for the Penn Machine Learning Benchmark data repository).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Load the data""")
    return


@app.cell
def _(fetch_data):
    # Load data
    data = fetch_data("adult")
    num_cols = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = [
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "target",
    ]
    data
    return cat_cols, data, num_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Gaussian Mixture Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Create and Train the GMM (or load if exists)""")
    return


@app.cell
def _(MODELS_DIR, RegularSynthesizer, cat_cols, data, num_cols, os):
    gmm_model_path = os.path.join(MODELS_DIR, "adult_gmm_model.pkl")

    if os.path.exists(gmm_model_path):
        # Load existing model
        gmm_synth = RegularSynthesizer.load(gmm_model_path)
    else:
        # Train new model
        gmm_synth = RegularSynthesizer(modelname="fast")
        gmm_synth.fit(data=data, num_cols=num_cols, cat_cols=cat_cols)
        gmm_synth.save(gmm_model_path)
    return (gmm_synth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Generate new synthetic data""")
    return


@app.cell
def _(SAMPLE_SIZE, gmm_synth):
    gmm_synth_data = gmm_synth.sample(SAMPLE_SIZE)
    gmm_synth_data
    return (gmm_synth_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Conditional Tabular Generative Adversarial Network""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Define model and training parameters""")
    return


@app.cell
def _(ModelParameters, TrainParameters):
    # Defining the training parameters
    batch_size = 500
    epochs = 500 + 1
    learning_rate = 2e-4
    beta_1 = 0.5
    beta_2 = 0.9

    ctgan_args = ModelParameters(
        batch_size=batch_size, lr=learning_rate, betas=(beta_1, beta_2)
    )

    train_args = TrainParameters(epochs=epochs)
    return ctgan_args, train_args


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Create and Train the CTGAN (or load if exists)""")
    return


@app.cell
def _(
    MODELS_DIR,
    RegularSynthesizer,
    cat_cols,
    ctgan_args,
    data,
    num_cols,
    os,
    train_args,
):
    model_path = os.path.join(MODELS_DIR, "adult_ctgan_model.pkl")

    if os.path.exists(model_path):
        # Load existing model
        ctgan_synth = RegularSynthesizer.load(model_path)
    else:
        # Train new model
        ctgan_synth = RegularSynthesizer(modelname="ctgan", model_parameters=ctgan_args)
        ctgan_synth.fit(
            data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols
        )
        ctgan_synth.save(model_path)
    return (ctgan_synth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Generate new synthetic data""")
    return


@app.cell
def _(SAMPLE_SIZE, ctgan_synth):
    ctgan_synth_data = ctgan_synth.sample(SAMPLE_SIZE)
    ctgan_synth_data
    return (ctgan_synth_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Distribution Comparison: Original vs GMM vs CTGAN""")
    return


@app.cell
def _(
    SAMPLE_SIZE,
    alt,
    cat_cols,
    ctgan_synth_data,
    data,
    gmm_synth_data,
    mo,
    num_cols,
    prepare_categorical_data,
    prepare_numeric_data,
):
    # Color scale for consistent styling
    color_scale = alt.Scale(
        domain=["Original", "GMM", "CTGAN"],
        range=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )

    # Create tabs for each column
    column_tabs = {}

    # Process numeric columns - use violin plots
    for col in num_cols:
        combined = prepare_numeric_data(
            data, gmm_synth_data, ctgan_synth_data, col, SAMPLE_SIZE
        )

        chart = (
            alt.Chart(combined)
            .transform_density(density=col, as_=[col, "density"], groupby=["Model"])
            .mark_area(orient="horizontal")
            .encode(
                alt.X(
                    "density:Q",
                    stack="center",
                    impute=None,
                    title=None,
                    axis=alt.Axis(labels=False, ticks=False, grid=False),
                ),
                alt.Y(f"{col}:Q", title=col),
                alt.Color("Model:N", scale=color_scale),
                alt.Column(
                    "Model:N",
                    header=alt.Header(titleOrient="bottom"),
                    sort=["Original", "GMM", "CTGAN"],
                ),
                tooltip=[f"{col}:Q", "Model:N"],
            )
            .properties(
                width=150,
                height=400,
                title=alt.TitleParams(
                    text=f"Distribution Comparison: {col}", anchor="middle"
                ),
            )
        )
        column_tabs[col] = chart

    # Process categorical columns - bar charts
    for col in cat_cols:
        combined = prepare_categorical_data(
            data, gmm_synth_data, ctgan_synth_data, col, SAMPLE_SIZE
        )

        chart = (
            alt.Chart(combined)
            .mark_bar()
            .encode(
                alt.X("count:Q", title="Count"),
                alt.Y(f"{col}:N", title=col, sort="ascending"),
                alt.Color("Model:N", scale=color_scale),
                alt.Column(
                    "Model:N",
                    header=alt.Header(titleOrient="bottom"),
                    sort=["Original", "GMM", "CTGAN"],
                ),
                tooltip=[col, "count:Q", "Model:N"],
            )
            .properties(
                width=150,
                height=400,
                title=alt.TitleParams(
                    text=f"Distribution Comparison: {col}", anchor="middle"
                ),
            )
        )
        column_tabs[col] = chart

    # Create tabs using marimo
    mo.ui.tabs(column_tabs)
    return


if __name__ == "__main__":
    app.run()
