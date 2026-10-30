"""
Program 6 : Named Entity Recognition (NER) using spaCy                     [CO1]
Aim       : Detect and classify real-world entities (person, organisation, place, date, money ...).
Algorithm : 1) load en_core_web_sm  2) run pipeline  3) read doc.ents  4) explain labels
            5) count entities  6) save displaCy visualisation (HTML)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from collections import Counter
from common import banner, load_spacy, OUTPUT_DIR
import spacy
from spacy import displacy

text = ("Sundar Pichai, the CEO of Google, announced on 12 March 2025 that the company will invest $10 billion "
        "in India. Meanwhile, Infosys opened a new campus in Bengaluru, and Prime Minister Narendra Modi "
        "visited the Noida Institute of Engineering and Technology near Delhi last Monday. "
        "Tata Motors sold about 50,000 cars in Mumbai in 2024.")
nlp = load_spacy()
doc = nlp(text)

banner("Entities found")
print(f"{'ENTITY':<45}{'LABEL':<10}{'MEANING'}")
for ent in doc.ents:
    print(f"{ent.text:<45}{ent.label_:<10}{spacy.explain(ent.label_)}")

banner("Entity counts by label")
for label, n in Counter(e.label_ for e in doc.ents).most_common():
    print(f"{label:<10}{n}")

banner("Entities grouped by type")
groups = {}
for e in doc.ents:
    groups.setdefault(e.label_, []).append(e.text)
for k, v in groups.items():
    print(f"{k:<10}{v}")

banner("IOB view of first 15 tokens")
print([(t.text, t.ent_iob_, t.ent_type_) for t in doc[:15]])

html = displacy.render(doc, style="ent", page=True)
out = OUTPUT_DIR / "06_ner_visualisation.html"
out.write_text(html, encoding="utf-8")
print("\nVisualisation saved to", out, "(open it in a browser)")
