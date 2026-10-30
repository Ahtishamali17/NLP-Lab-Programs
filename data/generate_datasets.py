"""Re-creates data/reviews.csv and data/news.csv (seeded, so results are reproducible)."""
import random, csv
from pathlib import Path
rng = random.Random(42)
here = Path(__file__).parent

# ---------------- reviews ----------------
aspects = ["battery life", "camera", "price", "delivery", "screen", "sound quality",
           "customer service", "build quality"]
products = ["phone", "laptop", "headphones", "smartwatch"]
pos = ["The {a} is excellent and I love it.", "Amazing {a}, totally worth the money.",
       "Very happy with the {a}, it works perfectly.", "Great {a} and really impressive."]
neg = ["The {a} is terrible and disappointing.", "Poor {a}, a complete waste of money.",
       "Very unhappy with the {a}, it stopped working.", "Worst {a} I have ever seen."]
rows = []
for i in range(80):
    label = "positive" if i % 2 == 0 else "negative"
    bank = pos if label == "positive" else neg
    a1, a2 = rng.sample(aspects, 2)
    p = rng.choice(products)
    text = f"I bought this {p}. " + rng.choice(bank).format(a=a1) + " " + rng.choice(bank).format(a=a2)
    rows.append((text, label))
mixed = [
 ("The camera is excellent but the battery life is terrible. Overall I am disappointed.", "negative"),
 ("Delivery was slow and the packaging was poor, however the screen is amazing and I love the phone.", "positive"),
 ("Great sound quality and fantastic build quality, but the price is a bit high. Still happy.", "positive"),
 ("The customer service was rude and unhelpful. The laptop stopped working in a week.", "negative"),
 ("Battery life is wonderful and the camera is superb. Delivery was fast. Highly recommended!", "positive"),
 ("Horrible screen, awful delivery, and the price is not worth it. I want a refund.", "negative"),
 ("Good price and decent camera, the delivery was on time. Satisfied with the purchase.", "positive"),
 ("The smartwatch has poor battery life and the build quality feels cheap. Very bad.", "negative"),
]
rows += mixed
rng.shuffle(rows)
with open(here / "reviews.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["review", "sentiment"]); w.writerows(rows)

# ---------------- news ----------------
bank = {
 "sports": (["The cricket team", "The football club", "The tennis champion", "The Olympic athlete", "The hockey squad"],
            ["won the championship match", "scored a late goal in the final", "broke the national record",
             "defeated rivals in the tournament", "secured a place in the league finals"],
            ["after a thrilling game", "in front of a cheering stadium crowd", "under the new coach", "during the season opener"]),
 "technology": (["The software company", "The AI startup", "A leading chipmaker", "The cloud provider", "The research lab"],
                ["launched a new machine learning platform", "unveiled a faster processor", "released an open source language model",
                 "announced a cybersecurity update for its servers", "introduced a smartphone with an improved camera"],
                ["at the developer conference", "for enterprise customers", "to improve data privacy", "using neural network technology"]),
 "health": (["The hospital", "Doctors", "The health ministry", "Researchers", "A medical team"],
            ["reported a decline in diabetes cases", "developed a new vaccine for infection", "recommended regular exercise and a balanced diet",
             "started a cancer screening program", "treated patients with a new therapy"],
            ["during the clinical trial", "to improve public health", "across rural clinics", "according to the medical journal"]),
 "politics": (["The prime minister", "The parliament", "The opposition party", "The election commission", "The government"],
              ["passed a new bill on taxation", "announced the election schedule", "debated the national budget policy",
               "signed a trade agreement with neighbouring countries", "proposed reforms in the constitution"],
              ["after a heated session", "ahead of the general election", "amid public protests", "following the cabinet meeting"]),
 "business": (["The stock market", "A major bank", "The retail company", "The startup", "The central bank"],
              ["reported record quarterly profits", "raised interest rates to control inflation", "acquired a rival firm in a billion dollar deal",
               "announced plans to expand exports", "saw shares rise sharply"],
              ["as investors gained confidence", "in the third quarter", "despite global uncertainty", "according to the annual report"]),
}
rows = []
for cat, (s, v, t) in bank.items():
    seen = set()
    while len(seen) < 12:
        seen.add(f"{rng.choice(s)} {rng.choice(v)} {rng.choice(t)}.")
    rows += [(x, cat) for x in sorted(seen)]
rng.shuffle(rows)
with open(here / "news.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["text", "category"]); w.writerows(rows)
print("datasets generated")
