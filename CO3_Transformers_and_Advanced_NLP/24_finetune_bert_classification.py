"""
Program 24 : Fine-Tuning BERT for text classification using Hugging Face  [CO3]
Aim        : Adapt a pre-trained transformer to our sentiment dataset (transfer learning).
Algorithm  : 1) load tokenizer + AutoModelForSequenceClassification (adds a fresh classification head)
             2) tokenize reviews (padding / truncation)  3) PyTorch DataLoader
             4) AdamW + linear warm-up scheduler  5) train for a few epochs  6) evaluate, predict, save
Install    : pip install torch transformers scikit-learn
Usage      : python 24_finetune_bert_classification.py                      (fast: distilbert-base-uncased)
             python 24_finetune_bert_classification.py --model bert-base-uncased --epochs 4
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from common import banner, load_reviews, OUTPUT_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilbert-base-uncased")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-5)
args = ap.parse_args()

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| model:", args.model)

df = load_reviews()
labels = ["negative", "positive"]
y = df.sentiment.map({l: i for i, l in enumerate(labels)}).values
X_train, X_test, y_train, y_test = train_test_split(df.review.tolist(), y, test_size=0.25, random_state=42, stratify=y)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(texts, padding="max_length", truncation=True, max_length=48, return_tensors="pt")
        self.labels = torch.tensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item

train_dl = DataLoader(ReviewDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
test_dl = DataLoader(ReviewDataset(X_test, y_test), batch_size=16)

optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps = len(train_dl) * args.epochs
sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)

def evaluate():
    model.eval(); preds = []
    with torch.no_grad():
        for b in test_dl:
            b = {k: v.to(device) for k, v in b.items()}
            preds += model(**{k: v for k, v in b.items() if k != "labels"}).logits.argmax(-1).cpu().tolist()
    return preds

banner("Fine-tuning")
for epoch in range(1, args.epochs + 1):
    model.train(); total = 0
    for b in train_dl:
        b = {k: v.to(device) for k, v in b.items()}
        loss = model(**b).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step(); sched.step(); optim.zero_grad(); total += loss.item()
    acc = accuracy_score(y_test, evaluate())
    print(f"epoch {epoch}/{args.epochs}   train loss = {total/len(train_dl):.4f}   test accuracy = {acc:.3f}")

banner("Final evaluation")
print(classification_report(y_test, evaluate(), target_names=labels, zero_division=0))

banner("Predictions on new reviews")
model.eval()
for s in ["The camera is excellent and I love it", "Worst battery, complete waste of money", "Delivery was late but the screen is wonderful"]:
    enc = tokenizer(s, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        p = torch.softmax(model(**enc).logits, -1)[0].cpu().numpy()
    print(f"{labels[p.argmax()]:<9} (confidence {p.max():.2f})  {s}")

save_dir = OUTPUT_DIR / "24_finetuned_model"
model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
print("\nFine-tuned model saved to", save_dir)
