#!/usr/bin/env python3
"""
Quick setup script to download required models for offline Presidio use.
Run this ONCE with internet connection before going offline.
"""

import subprocess
import sys


def run_command(description, command):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return False


def download_spacy_models():
    """Download required spaCy models."""
    models = ["en_core_web_sm", "en_core_web_lg"]
    for model in models:
        if not run_command(
            f"Downloading spaCy model: {model}",
            f"{sys.executable} -m spacy download {model}"
        ):
            return False
    return True


def download_huggingface_models():
    """Pre-download HuggingFace models to cache."""
    print(f"\n{'='*60}")
    print(f"📦 Downloading HuggingFace models (this may take a while)...")
    print(f"{'='*60}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        models = [
            "obi/deid_roberta_i2b2",
            "StanfordAIMI/stanford-deidentifier-base"
        ]
        
        for model_name in models:
            print(f"\n  → Downloading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            print(f"  ✅ {model_name} cached successfully")
        
        print(f"\n✅ All HuggingFace models downloaded!")
        return True
    except Exception as e:
        print(f"❌ Failed to download HuggingFace models: {e}")
        return False


def download_flair_models():
    """Pre-download Flair models."""
    print(f"\n{'='*60}")
    print(f"📦 Downloading Flair NER model...")
    print(f"{'='*60}")
    
    try:
        from flair.models import SequenceTagger
        tagger = SequenceTagger.load("ner-large")
        print(f"✅ Flair model downloaded!")
        return True
    except Exception as e:
        print(f"❌ Failed to download Flair model: {e}")
        return False


def download_stanza_models():
    """Pre-download Stanza models."""
    print(f"\n{'='*60}")
    print(f"📦 Downloading Stanza English model...")
    print(f"{'='*60}")
    
    try:
        import stanza
        stanza.download('en')
        print(f"✅ Stanza model downloaded!")
        return True
    except Exception as e:
        print(f"❌ Failed to download Stanza model: {e}")
        return False


def main():
    """Main setup function."""
    print("""
╔════════════════════════════════════════════════════════════╗
║        Presidio Offline Setup - Model Downloader           ║
║                                                            ║
║  This will download all required models for offline use    ║
║  Make sure you have an active internet connection!         ║
╚════════════════════════════════════════════════════════════╝
""")
    
    input("Press ENTER to continue (or Ctrl+C to cancel)... ")
    
    steps = [
        ("SpaCy Models", download_spacy_models),
        ("HuggingFace Models", download_huggingface_models),
        ("Flair Models", download_flair_models),
        ("Stanza Models", download_stanza_models),
    ]
    
    results = {}
    for name, func in steps:
        results[name] = func()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SETUP SUMMARY")
    print(f"{'='*60}")
    
    all_success = True
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {name}")
        if not success:
            all_success = False
    
    print(f"{'='*60}")
    
    if all_success:
        print("""
✅ All models downloaded successfully!

You can now run the app OFFLINE:

  uv run streamlit run presidio_streamlit.py

The app will work without any internet connection.
""")
        return 0
    else:
        print("""
⚠️  Some models failed to download.

Please check the error messages above and:
1. Ensure you have an active internet connection
2. Make sure all dependencies are installed: uv pip install -r requirements.txt
3. Try running this script again
""")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
