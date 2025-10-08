import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Tabular Synthetic Data Generation with GMM, WGAN-GP, DRAGAN, and CTGAN
    - This notebook demonstrates synthetic data generation using four different methods from [ydata-synthetic](https://github.com/ydataai/ydata-synthetic):
      - **GMM** ([Gaussian Mixture Model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)) for generating synthetic tabular data
      - **WGAN-GP** ([Wasserstein GAN with Gradient Penalty](https://arxiv.org/abs/1704.00028)) for improved GAN training stability
      - **DRAGAN** ([Deep Regret Analytic GAN](https://arxiv.org/abs/1705.07215)) for mode collapse prevention and stable training
      - **CTGAN** ([Conditional Tabular GAN](https://arxiv.org/pdf/1907.00503.pdf)) specifically designed for tabular data with numeric and categorical features
    """
    )
    return


@app.cell
def _():
    import os
    import json
    import contextlib
    import io
    import re
    import requests
    import yaml


    import altair as alt
    import marimo as mo
    import pandas as pd
    from pmlb import fetch_data
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer

    # Path to models directory
    MODELS_DIR = "models"

    # Sample size for visualization
    SAMPLE_SIZE = 1000
    return (
        MODELS_DIR,
        ModelParameters,
        RegularSynthesizer,
        SAMPLE_SIZE,
        TrainParameters,
        alt,
        contextlib,
        fetch_data,
        io,
        json,
        mo,
        os,
        pd,
        re,
        requests,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Dataset

    - The data used is the [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income) which we will fetch by importing the `pmlb` library (a wrapper for the Penn Machine Learning Benchmark data repository).
    """
    )
    return


@app.cell
def _(fetch_data, mo):
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
    mo.vstack(
        [
            mo.md("### Load the data"),
            data,
        ]
    )
    return cat_cols, data, num_cols


@app.cell
def _(contextlib, io, json, os, re):
    def parse_training_output(output_text, model_type):
        """Parse training output to extract losses."""
        losses = {"epoch": [], "generator_loss": [], "critic_loss": []}

        # Different patterns for different models
        if model_type == "ctgan":
            # Pattern: Epoch: 0 | critic_loss: -0.123 | generator_loss: 0.456
            pattern = r"Epoch:\s*(\d+)\s*\|\s*critic_loss:\s*([-\d.eE+]+)\s*\|\s*generator_loss:\s*([-\d.eE+]+)"
        elif model_type in ["wgangp", "dragan"]:
            # Pattern: Epoch: 0 | disc_loss: 0.123 | gen_loss: 0.456
            pattern = r"Epoch:\s*(\d+)\s*\|\s*disc_loss:\s*([-\d.eE+]+)\s*\|\s*gen_loss:\s*([-\d.eE+]+)"
        else:
            return losses

        for match in re.finditer(pattern, output_text):
            losses["epoch"].append(int(match.group(1)))
            if model_type == "ctgan":
                losses["critic_loss"].append(float(match.group(2)))
                losses["generator_loss"].append(float(match.group(3)))
            else:
                losses["critic_loss"].append(float(match.group(2)))
                losses["generator_loss"].append(float(match.group(3)))

        return losses

    def train_with_loss_tracking(synth, model_name, model_type, **fit_kwargs):
        """Train model and capture loss output."""
        # Capture stdout during training
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            synth.fit(**fit_kwargs)

        # Parse the captured output
        output = f.getvalue()
        losses = parse_training_output(output, model_type)

        # Save losses to JSON
        loss_path = os.path.join("models", f"{model_name}_losses.json")
        with open(loss_path, "w") as file:
            json.dump(losses, file)

        return losses

    def load_losses(model_name):
        """Load losses from JSON file."""
        loss_path = os.path.join("models", f"{model_name}_losses.json")
        if os.path.exists(loss_path):
            with open(loss_path, "r") as file:
                return json.load(file)
        return None
    return load_losses, train_with_loss_tracking


@app.cell
def _(requests, yaml):
    # Fetch the YAML file
    url = "https://raw.githubusercontent.com/EpistasisLab/pmlb/refs/heads/master/datasets/adult/metadata.yaml"
    response = requests.get(url)
    metadata = yaml.safe_load(response.text)

    metadata
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Gaussian Mixture Model""")
    return


@app.cell
def _(MODELS_DIR, RegularSynthesizer, cat_cols, data, mo, num_cols, os):
    gmm_model_path = os.path.join(MODELS_DIR, "adult_gmm_model.pkl")

    if os.path.exists(gmm_model_path):
        # Load existing model
        gmm_synth = RegularSynthesizer.load(gmm_model_path)
    else:
        # Train new model
        gmm_synth = RegularSynthesizer(modelname="fast")
        gmm_synth.fit(data=data, num_cols=num_cols, cat_cols=cat_cols)
        gmm_synth.save(gmm_model_path)

    mo.md("### Create and train the GMM (or load if exists)")
    return (gmm_synth,)


@app.cell
def _(SAMPLE_SIZE, gmm_synth, mo):
    gmm_synth_data = gmm_synth.sample(SAMPLE_SIZE)
    mo.vstack(
        [
            mo.md("### Generate new synthetic data"),
            gmm_synth_data,
        ]
    )
    return (gmm_synth_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Wasserstein Generative Adversarial Network with Gradient Penalty""")
    return


@app.cell
def _(ModelParameters, TrainParameters, mo):
    # Defining the training parameters for WGAN
    noise_dim = 128
    dim = 128
    wgan_batch_size = 500
    log_step = 100
    wgan_epochs = 40 + 1  # starts throwing NaNs before epoch 50
    wgan_learning_rate = [5e-4, 3e-3]
    wgan_beta_1 = 0.5
    wgan_beta_2 = 0.9

    wgan_args = ModelParameters(
        batch_size=wgan_batch_size,
        lr=wgan_learning_rate,
        betas=(wgan_beta_1, wgan_beta_2),
        noise_dim=noise_dim,
        layers_dim=dim,
    )

    wgan_train_args = TrainParameters(epochs=wgan_epochs, sample_interval=log_step)
    mo.md("### Define WGAN model and training parameters")
    return wgan_args, wgan_train_args


@app.cell
def _(
    MODELS_DIR,
    RegularSynthesizer,
    cat_cols,
    data,
    load_losses,
    mo,
    num_cols,
    os,
    train_with_loss_tracking,
    wgan_args,
    wgan_train_args,
):
    wgan_model_path = os.path.join(MODELS_DIR, "adult_wgangp_model.pkl")

    if os.path.exists(wgan_model_path):
        # Load existing model
        wgan_synth = RegularSynthesizer.load(wgan_model_path)
        wgan_losses = load_losses("adult_wgangp")
    else:
        # Train new model
        wgan_synth = RegularSynthesizer(
            modelname="wgangp", model_parameters=wgan_args, n_critic=2
        )
        wgan_losses = train_with_loss_tracking(
            wgan_synth,
            "adult_wgangp",
            "wgangp",
            data=data,
            train_arguments=wgan_train_args,
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        wgan_synth.save(wgan_model_path)

    mo.md("### Create and train the WGAN-GP (or load if exists)")
    return wgan_losses, wgan_synth


@app.cell
def _(SAMPLE_SIZE, alt, mo, pd, wgan_losses, wgan_synth):
    wgan_synth_data = wgan_synth.sample(SAMPLE_SIZE)

    # Create loss plot if losses are available
    wgan_loss_plot = None
    if wgan_losses and len(wgan_losses["epoch"]) > 0:
        wgan_loss_df = pd.DataFrame(wgan_losses)
        wgan_loss_long = pd.melt(
            wgan_loss_df,
            id_vars=["epoch"],
            value_vars=["generator_loss", "critic_loss"],
            var_name="loss_type",
            value_name="loss",
        )

        wgan_loss_plot = (
            alt.Chart(wgan_loss_long)
            .mark_line()
            .encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("loss:Q", title="Loss"),
                color=alt.Color(
                    "loss_type:N",
                    title="Loss Type",
                    scale=alt.Scale(
                        domain=["generator_loss", "critic_loss"],
                        range=["#1f77b4", "#ff7f0e"],
                    ),
                ),
                tooltip=["epoch:Q", "loss_type:N", "loss:Q"],
            )
            .properties(width=600, height=300, title="WGAN-GP Training Losses")
        )

    mo.vstack(
        [
            mo.md("### Generate new synthetic data with WGAN-GP"),
            wgan_synth_data,
            mo.md("### WGAN-GP Training Loss History") if wgan_loss_plot else mo.md(""),
            wgan_loss_plot if wgan_loss_plot else mo.md("*No loss history available. Train the model to see losses.*"),
        ]
    )
    return (wgan_synth_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Deep Regret Analytic Generative Adversarial Network""")
    return


@app.cell
def _(ModelParameters, TrainParameters, mo):
    # Defining the training parameters for DRAGAN
    dragan_noise_dim = 128
    dragan_dim = 128
    dragan_batch_size = 500
    dragan_log_step = 100
    dragan_epochs = 500 + 1
    dragan_learning_rate = 1e-5
    dragan_beta_1 = 0.5
    dragan_beta_2 = 0.9

    dragan_args = ModelParameters(
        batch_size=dragan_batch_size,
        lr=dragan_learning_rate,
        betas=(dragan_beta_1, dragan_beta_2),
        noise_dim=dragan_noise_dim,
        layers_dim=dragan_dim,
    )

    dragan_train_args = TrainParameters(
        epochs=dragan_epochs, sample_interval=dragan_log_step
    )
    mo.md("### Define DRAGAN model and training parameters")
    return dragan_args, dragan_train_args


@app.cell
def _(
    MODELS_DIR,
    RegularSynthesizer,
    cat_cols,
    data,
    dragan_args,
    dragan_train_args,
    load_losses,
    mo,
    num_cols,
    os,
    train_with_loss_tracking,
):
    dragan_model_path = os.path.join(MODELS_DIR, "adult_dragan_model.pkl")

    if os.path.exists(dragan_model_path):
        # Load existing model
        dragan_synth = RegularSynthesizer.load(dragan_model_path)
        dragan_losses = load_losses("adult_dragan")
    else:
        # Train new model
        dragan_synth = RegularSynthesizer(
            modelname="dragan", model_parameters=dragan_args, n_discriminator=3
        )
        dragan_losses = train_with_loss_tracking(
            dragan_synth,
            "adult_dragan",
            "dragan",
            data=data,
            train_arguments=dragan_train_args,
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        dragan_synth.save(dragan_model_path)

    mo.md("### Create and train the DRAGAN (or load if exists)")
    return dragan_losses, dragan_synth


@app.cell
def _(SAMPLE_SIZE, alt, dragan_losses, dragan_synth, mo, pd):
    dragan_synth_data = dragan_synth.sample(SAMPLE_SIZE)

    # Create loss plot if losses are available
    dragan_loss_plot = None
    if dragan_losses and len(dragan_losses["epoch"]) > 0:
        dragan_loss_df = pd.DataFrame(dragan_losses)
        dragan_loss_long = pd.melt(
            dragan_loss_df,
            id_vars=["epoch"],
            value_vars=["generator_loss", "critic_loss"],
            var_name="loss_type",
            value_name="loss",
        )

        dragan_loss_plot = (
            alt.Chart(dragan_loss_long)
            .mark_line()
            .encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("loss:Q", title="Loss"),
                color=alt.Color(
                    "loss_type:N",
                    title="Loss Type",
                    scale=alt.Scale(
                        domain=["generator_loss", "critic_loss"],
                        range=["#1f77b4", "#ff7f0e"],
                    ),
                ),
                tooltip=["epoch:Q", "loss_type:N", "loss:Q"],
            )
            .properties(width=600, height=300, title="DRAGAN Training Losses")
        )

    mo.vstack(
        [
            mo.md("### Generate new synthetic data with DRAGAN"),
            dragan_synth_data,
            mo.md("### DRAGAN Training Loss History") if dragan_loss_plot else mo.md(""),
            dragan_loss_plot if dragan_loss_plot else mo.md("*No loss history available. Train the model to see losses.*"),
        ]
    )
    return (dragan_synth_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Conditional Tabular Generative Adversarial Network""")
    return


@app.cell
def _(ModelParameters, TrainParameters, mo):
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
    mo.md("### Define model and training parameters")
    return ctgan_args, train_args


@app.cell
def _(
    MODELS_DIR,
    RegularSynthesizer,
    cat_cols,
    ctgan_args,
    data,
    load_losses,
    mo,
    num_cols,
    os,
    train_args,
    train_with_loss_tracking,
):
    model_path = os.path.join(MODELS_DIR, "adult_ctgan_model.pkl")

    if os.path.exists(model_path):
        # Load existing model
        ctgan_synth = RegularSynthesizer.load(model_path)
        ctgan_losses = load_losses("adult_ctgan")
    else:
        # Train new model
        ctgan_synth = RegularSynthesizer(modelname="ctgan", model_parameters=ctgan_args)
        ctgan_losses = train_with_loss_tracking(
            ctgan_synth,
            "adult_ctgan",
            "ctgan",
            data=data,
            train_arguments=train_args,
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        ctgan_synth.save(model_path)

    mo.md("### Create and train the CTGAN (or load if exists)")
    return ctgan_losses, ctgan_synth


@app.cell
def _(SAMPLE_SIZE, alt, ctgan_losses, ctgan_synth, mo, pd):
    ctgan_synth_data = ctgan_synth.sample(SAMPLE_SIZE)

    # Create loss plot if losses are available
    ctgan_loss_plot = None
    if ctgan_losses and len(ctgan_losses["epoch"]) > 0:
        ctgan_loss_df = pd.DataFrame(ctgan_losses)
        ctgan_loss_long = pd.melt(
            ctgan_loss_df,
            id_vars=["epoch"],
            value_vars=["generator_loss", "critic_loss"],
            var_name="loss_type",
            value_name="loss",
        )

        ctgan_loss_plot = (
            alt.Chart(ctgan_loss_long)
            .mark_line()
            .encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("loss:Q", title="Loss"),
                color=alt.Color(
                    "loss_type:N",
                    title="Loss Type",
                    scale=alt.Scale(
                        domain=["generator_loss", "critic_loss"],
                        range=["#1f77b4", "#ff7f0e"],
                    ),
                ),
                tooltip=["epoch:Q", "loss_type:N", "loss:Q"],
            )
            .properties(width=600, height=300, title="CTGAN Training Losses")
        )

    mo.vstack(
        [
            mo.md("### Generate new synthetic data with CTGAN"),
            ctgan_synth_data,
            mo.md("### CTGAN Training Loss History") if ctgan_loss_plot else mo.md(""),
            ctgan_loss_plot if ctgan_loss_plot else mo.md("*No loss history available. Train the model to see losses.*"),
        ]
    )
    return (ctgan_synth_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Distribution comparison: Original vs GMM vs WGAN vs DRAGAN vs CTGAN""")
    return


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
        wgan_data: pd.DataFrame,
        dragan_data: pd.DataFrame,
        ctgan_data: pd.DataFrame,
        col: str,
        sample_size: int,
    ) -> pd.DataFrame:
        """Prepare combined data for numeric column visualization."""
        original_sample = sample_column(data, col, sample_size)
        gmm_sample = sample_column(gmm_data, col, sample_size)
        wgan_sample = sample_column(wgan_data, col, sample_size)
        dragan_sample = sample_column(dragan_data, col, sample_size)
        ctgan_sample = sample_column(ctgan_data, col, sample_size)

        original_df = pd.DataFrame({col: original_sample, "Model": "Original"})
        gmm_df = pd.DataFrame({col: gmm_sample, "Model": "GMM"})
        wgan_df = pd.DataFrame({col: wgan_sample, "Model": "WGAN"})
        dragan_df = pd.DataFrame({col: dragan_sample, "Model": "DRAGAN"})
        ctgan_df = pd.DataFrame({col: ctgan_sample, "Model": "CTGAN"})

        return pd.concat([original_df, gmm_df, wgan_df, dragan_df, ctgan_df])

    def prepare_categorical_data(
        data: pd.DataFrame,
        gmm_data: pd.DataFrame,
        wgan_data: pd.DataFrame,
        dragan_data: pd.DataFrame,
        ctgan_data: pd.DataFrame,
        col: str,
        sample_size: int,
    ) -> pd.DataFrame:
        """Prepare combined data for categorical column visualization."""
        original_sample = sample_column(data, col, sample_size)
        gmm_sample = sample_column(gmm_data, col, sample_size)
        wgan_sample = sample_column(wgan_data, col, sample_size)
        dragan_sample = sample_column(dragan_data, col, sample_size)
        ctgan_sample = sample_column(ctgan_data, col, sample_size)

        # Get value counts for each dataset
        original_counts = original_sample.value_counts().reset_index()
        original_counts.columns = [col, "count"]
        original_counts["Model"] = "Original"

        gmm_counts = gmm_sample.value_counts().reset_index()
        gmm_counts.columns = [col, "count"]
        gmm_counts["Model"] = "GMM"

        wgan_counts = wgan_sample.value_counts().reset_index()
        wgan_counts.columns = [col, "count"]
        wgan_counts["Model"] = "WGAN"

        dragan_counts = dragan_sample.value_counts().reset_index()
        dragan_counts.columns = [col, "count"]
        dragan_counts["Model"] = "DRAGAN"

        ctgan_counts = ctgan_sample.value_counts().reset_index()
        ctgan_counts.columns = [col, "count"]
        ctgan_counts["Model"] = "CTGAN"

        return pd.concat(
            [original_counts, gmm_counts, wgan_counts, dragan_counts, ctgan_counts]
        )
    return prepare_categorical_data, prepare_numeric_data


@app.cell
def _(
    SAMPLE_SIZE,
    alt,
    cat_cols,
    ctgan_synth_data,
    data,
    dragan_synth_data,
    gmm_synth_data,
    mo,
    num_cols,
    prepare_categorical_data,
    prepare_numeric_data,
    wgan_synth_data,
):
    # Color scale for consistent styling
    color_scale = alt.Scale(
        domain=["Original", "GMM", "WGAN", "DRAGAN", "CTGAN"],
        range=["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c"],
    )

    # Create tabs for each column
    column_tabs = {}

    # Process numeric columns - use violin plots
    for col in num_cols:
        combined = prepare_numeric_data(
            data,
            gmm_synth_data,
            wgan_synth_data,
            dragan_synth_data,
            ctgan_synth_data,
            col,
            SAMPLE_SIZE,
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
                    sort=["Original", "GMM", "WGAN", "DRAGAN", "CTGAN"],
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
            data,
            gmm_synth_data,
            wgan_synth_data,
            dragan_synth_data,
            ctgan_synth_data,
            col,
            SAMPLE_SIZE,
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
                    sort=["Original", "GMM", "WGAN", "DRAGAN", "CTGAN"],
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
