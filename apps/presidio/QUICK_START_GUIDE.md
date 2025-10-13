# How to Use Presidio's Streamlit App Offline

## Setup (One-Time, Requires Internet)

1. After setting up the virtual environment and downloading package dependencies (see [README](../../README.md)), navigate to the app directory:

   ```bash
   cd apps/presidio
   ```

2. Install app dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

3. Download all NLP models:

   ```bash
   uv run python setup_models.py
   ```

   This downloads ~4-5 GB of models. Wait for completion.

## Running the App (Works Offline After Setup)

```bash
uv run streamlit run presidio_streamlit.py
```

The app will open in your browser at `http://localhost:8501` (or similar port).

## Using the App

1. Select an NLP model from the dropdown
2. Enter or paste the text for de-identification
3. Choose de-identification method (redact, replace, mask, etc.)
4. Review the de-identified output

## Available Models

- **spaCy/en_core_web_lg** - [Model Info](https://spacy.io/models/en#en_core_web_lg)
- **flair/ner-english-large** - [Model Info](https://huggingface.co/flair/ner-english-large)
- **HuggingFace/obi/deid_roberta_i2b2** - [Model Info](https://huggingface.co/obi/deid_roberta_i2b2)
- **HuggingFace/StanfordAIMI/stanford-deidentifier-base** - [Model Info](https://huggingface.co/StanfordAIMI/stanford-deidentifier-base)
- **stanza/en** - [Model Info](https://stanfordnlp.github.io/stanza/)
