"""
Program 20 : Information Extraction (IE) from structured / unstructured documents   [CO2]
Aim        : Turn text into structured records.
Algorithm  : A) UNSTRUCTURED text  - regex patterns (email, phone, date, money, URL) + spaCy NER
                                      + relation triples (subject, verb, object) from the dependency parse
             B) STRUCTURED / semi-structured - parse JSON, CSV and "Key: Value" forms into one record
             C) save everything as JSON
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import csv, io, json, re
from common import banner, load_spacy, OUTPUT_DIR

unstructured = """
Invoice No: INV-2025-0457 dated 14/03/2025.
Dear Ms. Ruchika Sharma, Noida Institute of Engineering and Technology purchased 25 laptops from Dell India
for Rs. 18,75,000 on 14 March 2025. The delivery will be made at Greater Noida by 20-03-2025.
For queries mail ruchika@niet.co.in or sales@dell.com, call +91-9466775555 or visit https://www.niet.co.in.
Sundar Pichai announced that Google will open a new office in Hyderabad.
"""

PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "phone": r"(?:\+91[-\s]?)?\b[6-9]\d{9}\b",
    "url":   r"https?://[^\s]+",
    "date":  r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b",
    "money": r"(?:Rs\.?|INR|\$|₹)\s?[\d,]+(?:\.\d+)?",
    "invoice_no": r"\bINV-\d{4}-\d+\b",
}

banner("A1. Regex based extraction")
regex_out = {k: re.findall(p, unstructured) for k, p in PATTERNS.items()}
for k, v in regex_out.items():
    print(f"{k:<11}: {v}")

banner("A2. Named entities (spaCy)")
nlp = load_spacy()
doc = nlp(unstructured)
entities = [(e.text, e.label_) for e in doc.ents]
print(entities)

banner("A3. Relation triples (subject, verb, object)")
triples = []
for sent in doc.sents:
    for t in sent:
        if t.pos_ == "VERB":
            subj = [c for c in t.children if c.dep_ in ("nsubj", "nsubjpass")]
            obj = [c for c in t.children if c.dep_ in ("dobj", "obj")]
            if subj and obj:
                triples.append((" ".join(w.text for w in subj[0].subtree if not w.is_space), t.lemma_, " ".join(w.text for w in obj[0].subtree)))
for tr in triples:
    print(tr)

banner("B. Structured / semi-structured sources")
json_src = '{"name": "Amit Verma", "course": "B.Tech AI", "year": 3, "skills": ["Python", "NLP"]}'
csv_src = "roll,name,marks\n101,Neha,88\n102,Ravi,74\n103,Pooja,91"
form_src = "Name: Priya Singh\nDepartment: Artificial Intelligence\nEmail: priya@niet.co.in\nCGPA: 8.9"
rec_json = json.loads(json_src)
rec_csv = list(csv.DictReader(io.StringIO(csv_src)))
rec_form = dict(re.findall(r"^(\w[\w ]*):\s*(.+)$", form_src, flags=re.M))
print("JSON  ->", rec_json); print("CSV   ->", rec_csv); print("FORM  ->", rec_form)
print("Average marks from CSV:", sum(int(r["marks"]) for r in rec_csv) / len(rec_csv))

result = {"regex": regex_out, "entities": entities, "triples": triples,
          "structured": {"json": rec_json, "csv": rec_csv, "form": rec_form}}
out = OUTPUT_DIR / "20_extracted_information.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nStructured result saved to", out)
