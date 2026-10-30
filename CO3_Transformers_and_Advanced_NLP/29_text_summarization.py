"""
Program 29 : Text Summarization using BART or T5                           [CO3]
Aim        : ABSTRACTIVE summarization - the model writes new sentences instead of copying them.
Algorithm  : 1) read data/sample_document.txt  2) tokenize (truncate to model limit)
             3) beam search generation  4) decode  5) compression ratio and ROUGE scores
Models     : sshleifer/distilbart-cnn-12-6 (small BART, default),  t5-small,
             optionally facebook/bart-large-cnn  (--big, ~1.6 GB)
Install    : pip install torch transformers sentencepiece rouge-score
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from common import banner, read_text, OUTPUT_DIR

text = " ".join(read_text("sample_document.txt").split())
print(f"Document length: {len(text.split())} words")

def summarize(model_name, text, prefix="", max_len=90, min_len=30):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
    enc = tok(prefix + text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = model.generate(**enc, num_beams=4, max_length=max_len, min_length=min_len,
                             length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)

models = [("BART (distilbart-cnn-12-6)", "sshleifer/distilbart-cnn-12-6", ""),
          ("T5-small", "t5-small", "summarize: ")]
if "--big" in sys.argv:
    models.insert(0, ("BART-large-cnn", "facebook/bart-large-cnn", ""))

summaries = {}
for label, name, prefix in models:
    banner(label)
    summaries[label] = summarize(name, text, prefix)
    s = summaries[label]
    print(s)
    print(f"\nSummary length: {len(s.split())} words | compression ratio: {len(text.split())/max(len(s.split()),1):.1f}x")

banner("ROUGE scores (summary vs. first 3 sentences of the document as a rough reference)")
try:
    from rouge_score import rouge_scorer
    ref = " ".join(text.split(". ")[:3])
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for label, s in summaries.items():
        r = sc.score(ref, s)
        print(f"{label:<28}" + "  ".join(f"{k}: F1={v.fmeasure:.2f}" for k, v in r.items()))
except ImportError:
    print("pip install rouge-score to enable ROUGE")

(OUTPUT_DIR / "29_summaries.txt").write_text("\n\n".join(f"[{k}]\n{v}" for k, v in summaries.items()), encoding="utf-8")
print("\nSaved to outputs/29_summaries.txt")
