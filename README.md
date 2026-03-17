# SynthData

A sandbox for synthetic data generation and evaluation.

It keeps forks of [`syntheval`](https://github.com/schneiderkamplab/syntheval) and [`synthcity`](https://github.com/vanderschaarlab/synthcity) as editable submodules to make it easy to test new features and bug fixes in those libraries. It also contains some early versions of apps, notebooks, and scripts for testing out different synthetic data generation and evaluation techniques.

## Quick Start

Clone the repo, initialize submodules, and install the main environment with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/childmindresearch/synthdata.git
cd synthdata
git submodule update --init --recursive
uv sync
```

`uv sync` installs the newer synthcity and syntheval workflow by default. Install optional extras when you need the older experiment tracks:

```bash
uv sync --extra ydata
uv sync --extra presidio
```

## Apps

- [`apps/presidio/presidio_streamlit.py`](apps/presidio/presidio_streamlit.py): Presidio's Streamlit app, modified for offline use. For the full version of the anonymizer, see [`anonymize-pii`](https://github.com/childmindresearch/anonymize-pii).

    See [PRESIDIO APP GUIDE](apps/presidio/PRESIDIO_APP_GUIDE.md) for details.

## Notebooks

- [`notebooks/ydata-test.py`](notebooks/ydata-test.py): Testing ydata-synthetic library for tabular data synthesis. To run using [`marimo`](https://github.com/marimo-team/marimo):

    ```bash
    uv run --extra ydata marimo run notebooks/ydata-test.py
    ```

- [`notebooks/test_hepatitis_data.ipynb`](notebooks/test_hepatitis_data.ipynb): Testing synthcity generators and syntheval & synthcity evaluations on the hepatitis dataset.

## Scripts

- [`scripts/markdown_parser.py`](scripts/markdown_parser.py): Early, monolithic version of the markdown parser. For the full version, see [`headhunter`](https://github.com/childmindresearch/headhunter).
