"""
Program 27 : Named Entity Recognition using Transformers (BERT)            [CO3]
Aim        : Token classification with a BERT model fine-tuned on CoNLL-2003 (dslim/bert-base-NER).
Algorithm  : 1) load tokenizer + AutoModelForTokenClassification  2) forward pass -> label for every word-piece
             3) merge word-pieces into entities (pipeline aggregation)  4) show IOB tags and entity table
             5) compare with spaCy NER (Program 6)
Install    : pip install torch transformers
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from common import banner, load_spacy

MODEL = "dslim/bert-base-NER"
texts = ["Sundar Pichai, the CEO of Google, visited Bengaluru and met Narendra Modi last Monday.",
         "Infosys and Tata Consultancy Services opened new offices in Noida near New Delhi.",
         "Ruchika teaches Natural Language Processing at the Noida Institute of Engineering and Technology."]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL).eval()

banner("Raw word-piece predictions (IOB tags) for sentence 1")
enc = tok(texts[0], return_tensors="pt")
with torch.no_grad():
    pred = model(**enc).logits.argmax(-1)[0]
for t, p in zip(tok.convert_ids_to_tokens(enc["input_ids"][0]), pred):
    print(f"{t:<14}{model.config.id2label[int(p)]}")

banner("Entities after merging word-pieces (aggregation_strategy='simple')")
ner = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple")
for t in texts:
    print("\n" + t)
    for e in ner(t):
        print(f"   {e['word']:<45}{e['entity_group']:<6} score={e['score']:.3f}")

banner("Comparison with spaCy (Program 6)")
nlp = load_spacy()
for t in texts[:2]:
    print("BERT :", [(e["word"], e["entity_group"]) for e in ner(t)])
    print("spaCy:", [(e.text, e.label_) for e in nlp(t).ents], "\n")
print("BERT labels: PER person, ORG organisation, LOC location, MISC miscellaneous.")
