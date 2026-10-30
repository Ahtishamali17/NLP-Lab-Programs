"""
Program 28 : Machine Translation using MarianMT or T5 (Hugging Face)       [CO3]
Aim        : Translate English into Hindi / French / German with pre-trained seq2seq transformers.
Algorithm  : 1) load tokenizer + AutoModelForSeq2SeqLM (encoder-decoder)  2) encode source sentences
             3) generate target tokens with beam search  4) decode  5) T5 uses a task prefix
             6) BLEU score against a reference translation
Install    : pip install torch transformers sentencepiece sacremoses
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from common import banner, ensure_nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

sentences = ["Natural language processing is fun.",
             "The students are learning machine translation in the lab.",
             "Please upload your programs to GitHub before the deadline."]

def translate(model_name, texts, prefix=""):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
    enc = tok([prefix + t for t in texts], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model.generate(**enc, num_beams=4, max_new_tokens=80)
    return tok.batch_decode(out, skip_special_tokens=True)

banner("MarianMT  (Helsinki-NLP opus-mt models, one model per language pair)")
marian = {"Hindi": "Helsinki-NLP/opus-mt-en-hi", "French": "Helsinki-NLP/opus-mt-en-fr", "German": "Helsinki-NLP/opus-mt-en-de"}
marian_out = {}
for lang, name in marian.items():
    try:
        marian_out[lang] = translate(name, sentences)
    except Exception as e:
        print(f"[skip] {lang}: {type(e).__name__}: {e}"); continue
    print(f"\n--- English -> {lang} ({name})")
    for s, t in zip(sentences, marian_out[lang]):
        print(f"  {s}\n  -> {t}")

banner("T5 (one multi-task model, task given as a text prefix)")
t5 = {"German": "translate English to German: ", "French": "translate English to French: "}
for lang, prefix in t5.items():
    out = translate("t5-small", sentences, prefix)
    print(f"\n--- T5-small English -> {lang}")
    for s, t in zip(sentences, out):
        print(f"  {s}\n  -> {t}")

banner("BLEU score (MarianMT en->fr against a human reference)")
ensure_nltk()
ref = "Le traitement du langage naturel est amusant."
if "French" in marian_out:
    hyp = marian_out["French"][0]
    smooth = SmoothingFunction().method1
    print("Reference :", ref); print("Hypothesis:", hyp)
    print("BLEU-4    :", round(sentence_bleu([ref.lower().split()], hyp.lower().split(), smoothing_function=smooth), 3))
