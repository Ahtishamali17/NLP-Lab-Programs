"""
One-time setup: python setup_resources.py [--models]
Downloads NLTK corpora, the spaCy English model and (with --models) the Hugging Face models used in Programs 24-30.
"""
import subprocess, sys

def main():
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng", "vader_lexicon", "universal_tagset"]:
        print("nltk:", pkg, nltk.download(pkg, quiet=True))
    print("spaCy model en_core_web_sm ...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    try:
        import gensim.downloader as api
        print("GloVe (glove-wiki-gigaword-50) ...")
        api.load("glove-wiki-gigaword-50")
    except Exception as e:
        print("GloVe download skipped:", e)
    if "--models" in sys.argv:
        from huggingface_hub import snapshot_download
        for m in ["distilbert-base-uncased", "bert-base-uncased", "gpt2", "dslim/bert-base-NER",
                  "Helsinki-NLP/opus-mt-en-hi", "Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-en-de",
                  "t5-small", "sshleifer/distilbart-cnn-12-6", "microsoft/DialoGPT-small"]:
            print("downloading", m); snapshot_download(m)
    print("\nSetup finished.")

if __name__ == "__main__":
    main()
