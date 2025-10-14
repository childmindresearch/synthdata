"""Streamlit app for Presidio - Modified for Offline Use."""
import logging
import os
import traceback

# Configure for offline operation - use cached models only
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
# Configure tldextract to use only its bundled snapshot (no internet download)
os.environ['TLDEXTRACT_CACHE'] = os.path.expanduser('~/.cache/python-tldextract')
# Disable Streamlit telemetry/usage statistics (prevents Fivetran webhook calls)
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_TELEMETRY_ENABLED'] = 'false'

# Configure tldextract before presidio imports it - use bundled snapshot only, no internet
import tldextract
# Monkey-patch the global extractor to use only bundled data (suffix_list_urls=())
# This prevents network calls to download the public suffix list
tldextract.TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())  # type: ignore

import pandas as pd
import streamlit as st
from annotated_text import annotated_text
from streamlit_tags import st_tags

from presidio_helpers import (
    get_supported_entities,
    analyze,
    anonymize,
    annotate,
    analyzer_engine,
)

st.set_page_config(
    page_title="Presidio demo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "https://microsoft.github.io/presidio/",
    },
)

logger = logging.getLogger("presidio-streamlit")
allow_other_models = os.getenv("ALLOW_OTHER_MODELS", False)

# Sidebar
st.sidebar.header(
    """
PII De-Identification with [Microsoft Presidio](https://microsoft.github.io/presidio/)
"""
)

model_help_text = """
    Select which Named Entity Recognition (NER) model to use for PII detection, in parallel to rule-based recognizers.
    All models run locally on your machine. Presidio supports multiple NER packages including spaCy, Huggingface, Stanza and Flair.
    """
st_ta_key = st_ta_endpoint = ""

model_list = [
    "spaCy/en_core_web_lg",
    "flair/ner-english-large",
    "HuggingFace/obi/deid_roberta_i2b2",
    "HuggingFace/StanfordAIMI/stanford-deidentifier-base",
    "stanza/en",
]

# Select model
st_model = st.sidebar.selectbox(
    "NER model package",
    model_list,
    index=2,
    help=model_help_text,
)

# Extract model package.
st_model_package = st_model.split("/")[0]

# Remove package prefix (if needed)
st_model = (
    st_model
    if st_model_package.lower() not in ("spacy", "stanza", "huggingface")
    else "/".join(st_model.split("/")[1:])
)

analyzer_params = (st_model_package, st_model, st_ta_key, st_ta_endpoint)
logger.debug(f"analyzer_params: {analyzer_params}")

st_operator = st.sidebar.selectbox(
    "De-identification approach",
    ["redact", "replace", "highlight", "mask", "hash", "encrypt"],
    index=1,
    help="""
    Select which manipulation to the text is requested after PII has been identified.\n
    - Redact: Completely remove the PII text\n
    - Replace: Replace the PII text with a constant, e.g. <PERSON>\n
    - Highlight: Shows the original text with PII highlighted in colors\n
    - Mask: Replaces a requested number of characters with an asterisk (or other mask character)\n
    - Hash: Replaces with the hash of the PII string\n
    - Encrypt: Replaces with an AES encryption of the PII string, allowing the process to be reversed
         """,
)
st_mask_char = "*"
st_number_of_chars = 15
st_encrypt_key = "WmZq4t7w!z%C&F)J"

logger.debug(f"st_operator: {st_operator}")

if st_operator == "mask":
    st_number_of_chars = st.sidebar.number_input(
        "number of chars", value=st_number_of_chars, min_value=0, max_value=100
    )
    st_mask_char = st.sidebar.text_input(
        "Mask character", value=st_mask_char, max_chars=1
    )
elif st_operator == "encrypt":
    st_encrypt_key = st.sidebar.text_input("AES key", value=st_encrypt_key)

st_threshold = st.sidebar.slider(
    label="Acceptance threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    help="Define the threshold for accepting a detection as PII. See more here: ",
)

st_return_decision_process = st.sidebar.checkbox(
    "Add analysis explanations to findings",
    value=False,
    help="Add the decision process to the output table. "
    "More information can be found here: https://microsoft.github.io/presidio/analyzer/decision_process/",
)

# Allow and deny lists
st_deny_allow_expander = st.sidebar.expander(
    "Allowlists and denylists",
    expanded=False,
)

with st_deny_allow_expander:
    st_allow_list = st_tags(
        label="Add words to the allowlist", text="Enter word and press enter."
    )
    st.caption(
        "Allowlists contain words that are not considered PII, but are detected as such."
    )

    st_deny_list = st_tags(
        label="Add words to the denylist", text="Enter word and press enter."
    )
    st.caption(
        "Denylists contain words that are considered PII, but are not detected as such."
    )

# Main panel
with st.expander("About this demo", expanded=False):
    st.info(
        """Presidio is an open source customizable framework for PII detection and de-identification.
        \n\n[Code](https://aka.ms/presidio) | 
        [Tutorial](https://microsoft.github.io/presidio/tutorial/) | 
        [Installation](https://microsoft.github.io/presidio/installation/) | 
        [FAQ](https://microsoft.github.io/presidio/faq/) |
        [Feedback](https://forms.office.com/r/9ufyYjfDaY) |"""
    )

    st.info(
        """
    Use this demo to:
    - Experiment with different local NLP models and packages (spaCy, Flair, HuggingFace, Stanza).
    - Explore the different de-identification options, including redaction, masking, encryption and more.
    - Configure allow and deny lists.
    
    This is a modified version that runs entirely offline (except for downloading models).
    [Visit Presidio's website](https://microsoft.github.io/presidio) for more info.
    """
    )

st.warning(
    """
    To minimize the risk of data leakage when processing sensitive information (e.g., PHI, PII), please follow these guidelines:
    - Disconnect the machine from the internet before loading or processing PHI.
    - When finished, close the browser tab and explicitly terminate the Streamlit process in the terminal before reconnecting the machine to any network.
    """,
    icon="⚠️",
)

analyzer_load_state = st.info("Starting Presidio analyzer...")

analyzer_load_state.empty()

# Read default text
demo_text_path = os.path.join(os.path.dirname(__file__), "demo_text.txt")
with open(demo_text_path) as f:
    demo_text = f.readlines()

# Create two columns for before and after
col1, col2 = st.columns(2)

# Before:
col1.subheader("Input")
st_text = col1.text_area(
    label="Enter text", value="".join(demo_text), height=400, key="text_input"
)

try:
    # Choose entities
    st_entities_expander = st.sidebar.expander("Choose entities to look for")
    st_entities = st_entities_expander.multiselect(
        label="Which entities to look for?",
        options=get_supported_entities(*analyzer_params),
        default=list(get_supported_entities(*analyzer_params)),
        help="Limit the list of PII entities detected. "
        "This list is dynamic and based on the NER model and registered recognizers. "
        "More information can be found here: https://microsoft.github.io/presidio/analyzer/adding_recognizers/",
    )

    # Before
    analyzer_load_state = st.info("Starting Presidio analyzer...")
    analyzer = analyzer_engine(*analyzer_params)
    analyzer_load_state.empty()

    st_analyze_results = analyze(
        *analyzer_params,
        text=st_text,
        entities=st_entities,
        language="en",
        score_threshold=st_threshold,
        return_decision_process=st_return_decision_process,
        allow_list=st_allow_list,
        deny_list=st_deny_list,
    )

    # After
    if st_operator != "highlight":
        with col2:
            st.subheader(f"Output")
            st_anonymize_results = anonymize(
                text=st_text,
                operator=st_operator,
                mask_char=st_mask_char,
                number_of_chars=st_number_of_chars,
                encrypt_key=st_encrypt_key,
                analyze_results=st_analyze_results,
            )
            st.text_area(
                label="De-identified", value=st_anonymize_results.text, height=400
            )
    else:
        st.subheader("Highlighted")
        annotated_tokens = annotate(text=st_text, analyze_results=st_analyze_results)
        # annotated_tokens
        annotated_text(*annotated_tokens)

    # table result
    st.subheader(
        "Findings"
        if not st_return_decision_process
        else "Findings with decision factors"
    )
    if st_analyze_results:
        df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
        df["text"] = [st_text[res.start : res.end] for res in st_analyze_results]

        df_subset = df[["entity_type", "text", "start", "end", "score"]].rename(
            {
                "entity_type": "Entity type",
                "text": "Text",
                "start": "Start",
                "end": "End",
                "score": "Confidence",
            },
            axis=1,
        )
        df_subset["Text"] = [st_text[res.start : res.end] for res in st_analyze_results]
        if st_return_decision_process:
            analysis_explanation_df = pd.DataFrame.from_records(
                [r.analysis_explanation.to_dict() for r in st_analyze_results]
            )
            df_subset = pd.concat([df_subset, analysis_explanation_df], axis=1)
        st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
    else:
        st.text("No findings")

except Exception as e:
    print(e)
    traceback.print_exc()
    st.error(e)
