# SynthData

A sandbox for synthetic data generation and evaluation.

## Quick Start

Clone the repo and set up a virtual environment using `uv`:

```bash
uv venv
```

Install dependencies:

```bash
uv sync
```

## List of Apps/Notebooks

- `apps/presidio/presidio_streamlit.py`: Presidio's Streamlit app, modified for offline use.

    See [QUICK START GUIDE](apps/presidio/QUICK_START_GUIDE.md) for details.

- `notebooks/ydata-test.py`: Testing ydata-synthetic library for tabular data synthesis. To edit using `marimo`:

    ```bash
    uv run marimo edit notebooks/ydata-test.py
    ```
