# How to Use Presidio's Streamlit App Offline

This Streamlit app is based on the [Huggingface Presidio demo](https://huggingface.co/spaces/presidio/presidio_demo) ([code for the app](https://github.com/microsoft/presidio/tree/main/docs/samples/python/streamlit)) and was modified so it can be used **fully offline** for PHI/PII processing without sending telemetry/usage data to Microsoft, Streamlit, HuggingFace, or other third parties.

## ⚠️ Security Warning for PHI/PII Processing

To minimize the risk of data leakage when processing sensitive information (e.g., PHI, PII), please follow these guidelines:

- **Disconnect the machine from the internet** before loading or processing PHI
- When finished, **close the browser tab and explicitly terminate the Streamlit process** in the terminal (Ctrl+C) before reconnecting the machine to any network

## Setup (One-Time, Requires Internet)

1. After setting up the virtual environment and downloading package dependencies (see [README](../../README.md)), navigate to the app directory:

   ```bash
   cd apps/presidio
   ```

2. Install app dependencies:

   ```bash
   uv add -r requirements.txt
   ```

3. Download all NLP models:

   ```bash
   uv run python setup_models.py
   ```

   This downloads ~4-5 GB of models. Wait for completion.

## Running the App (Works Offline After Setup)

**⚠️ For PHI/PII use: Disconnect from internet first!**

```bash
uv run streamlit run presidio_streamlit.py
```

> Note: The first time you run Streamlit it may prompt for your email. You can press Enter (leave the field blank) to skip this step.

The app will open in your browser at `http://localhost:8501` (or similar port).

## Using the App

1. **Verify you're offline** (if processing PHI/PII)
2. Select an NLP model from the dropdown
3. Enter or paste the text for de-identification
4. Choose de-identification method (redact, replace, mask, etc.)
5. Review the de-identified output
6. **When finished:** Close browser tab, press Ctrl+C in terminal to stop the app, then reconnect to the internet

## Available Models

- **spaCy/en_core_web_lg** - [Model Info](https://spacy.io/models/en#en_core_web_lg)
- **flair/ner-english-large** - [Model Info](https://huggingface.co/flair/ner-english-large)
- **HuggingFace/obi/deid_roberta_i2b2** - [Model Info](https://huggingface.co/obi/deid_roberta_i2b2)
- **HuggingFace/StanfordAIMI/stanford-deidentifier-base** - [Model Info](https://huggingface.co/StanfordAIMI/stanford-deidentifier-base)
- **stanza/en** - [Model Info](https://stanfordnlp.github.io/stanza/)
