"""
Program 19 : Opinion Mining on product / service reviews dataset           [CO2]
Aim        : Aspect-based opinion mining - find WHICH feature (battery, camera, price ...) people like or dislike.
Algorithm  : 1) split every review into sentences/clauses  2) detect aspect terms with a keyword lexicon
             3) score the opinion in that clause with VADER  4) extract opinion words (adjectives) with spaCy
             5) aggregate per aspect (positive %, negative %, mean score)  6) overall rating + chart
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import re
from collections import Counter, defaultdict
import pandas as pd
from common import banner, load_reviews, load_spacy, OUTPUT_DIR
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ASPECTS = {"battery": ["battery", "charge", "charging"], "camera": ["camera", "photo", "picture"],
           "price": ["price", "cost", "money", "expensive", "cheap"], "delivery": ["delivery", "shipping", "packaging"],
           "screen": ["screen", "display"], "sound": ["sound", "audio", "speaker"],
           "service": ["service", "support", "refund"], "build": ["build", "quality", "design"]}

df = load_reviews()
vader = SentimentIntensityAnalyzer()
nlp = load_spacy()
rows, opinion_words = [], defaultdict(Counter)

for rid, text in enumerate(df.review):
    for sent in re.split(r"(?<=[.!?])\s+", text):
        for clause in re.split(r",|\bbut\b|\bhowever\b", sent):       # a contrast word starts a new opinion
            c = clause.lower()
            for aspect, kws in ASPECTS.items():
                if any(re.search(rf"\b{k}", c) for k in kws):
                    score = vader.polarity_scores(clause)["compound"]
                    rows.append(dict(review_id=rid, aspect=aspect, clause=clause.strip(), score=score))
                    for t in nlp(clause):                                 # opinion words = adjectives
                        if t.pos_ == "ADJ":
                            opinion_words[aspect][t.lemma_.lower()] += 1

op = pd.DataFrame(rows)
print(f"{len(df)} reviews -> {len(op)} aspect opinions found")
print(op.head(8).to_string(index=False))

banner("Aspect level summary")
g = op.groupby("aspect").score
summary = pd.DataFrame({"mentions": g.size(),
                        "positive_%": g.apply(lambda s: round(100 * (s > 0.05).mean(), 1)),
                        "negative_%": g.apply(lambda s: round(100 * (s < -0.05).mean(), 1)),
                        "avg_score": g.mean().round(3)}).sort_values("avg_score", ascending=False)
print(summary)

banner("Text bar chart of average sentiment")
for a, row in summary.iterrows():
    n = int(abs(row.avg_score) * 30)
    print(f"{a:<9}{'+' * n if row.avg_score >= 0 else '-' * n} {row.avg_score:+.2f}")

banner("Opinion words used for each aspect")
for a in summary.index:
    print(f"{a:<9}", opinion_words[a].most_common(5))

banner("Overall verdict")
overall = op.score.mean()
print(f"Average opinion score = {overall:+.3f} ->", "POSITIVE" if overall > 0.05 else "NEGATIVE" if overall < -0.05 else "NEUTRAL")
print("Best aspect :", summary.index[0], "| Worst aspect:", summary.index[-1])

summary.to_csv(OUTPUT_DIR / "19_aspect_summary.csv")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ax = summary.avg_score.plot(kind="barh", color=["tab:green" if v > 0 else "tab:red" for v in summary.avg_score], figsize=(7, 4))
    ax.set_title("Aspect based sentiment"); ax.set_xlabel("average VADER score"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "19_aspect_sentiment.png", dpi=120)
    print("\nSaved outputs/19_aspect_summary.csv and outputs/19_aspect_sentiment.png")
except Exception as e:
    print("chart skipped:", e)
