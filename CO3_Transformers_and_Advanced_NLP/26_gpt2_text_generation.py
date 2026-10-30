"""
Program 26 : Text Generation using GPT-2 on a custom dataset               [CO3]
Aim        : Fine-tune GPT-2 (124M parameters) with the causal language-modelling objective on data/gpt2_custom.txt
             and compare the text it generates BEFORE and AFTER fine-tuning.
Algorithm  : 1) tokenize the custom text and cut it into blocks  2) labels = input ids (next-token prediction)
             3) train a few epochs with AdamW  4) generate with top-k / top-p / temperature sampling
Install    : pip install torch transformers
Usage      : python 26_gpt2_text_generation.py --epochs 8
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import banner, read_text, OUTPUT_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--lr", type=float, default=5e-5)
args = ap.parse_args()

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

prompts = ["The students in the NLP lab", "The teacher explains", "Every great project"]

def generate(prompt, n=1):
    model.eval()
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=35, do_sample=True, top_k=50, top_p=0.95, temperature=0.8,
                             num_return_sequences=n, pad_token_id=tok.eos_token_id)
    return [tok.decode(o, skip_special_tokens=True).replace("\n", " ") for o in out]

banner("BEFORE fine-tuning")
for p in prompts:
    torch.manual_seed(1); print(f"> {generate(p)[0]}\n")

# ---- build training blocks -------------------------------------------------------------
text = read_text("gpt2_custom.txt")
ids = tok(text + tok.eos_token, return_tensors="pt").input_ids[0]
blocks = [ids[i:i + args.block] for i in range(0, len(ids) - args.block, args.block // 2)]   # overlapping windows
print(f"Training text: {len(ids)} tokens -> {len(blocks)} blocks of {args.block} tokens")
dl = DataLoader(torch.stack(blocks), batch_size=4, shuffle=True)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
banner("Fine-tuning GPT-2")
for ep in range(1, args.epochs + 1):
    model.train(); tot = 0
    for batch in dl:
        batch = batch.to(device)
        loss = model(input_ids=batch, labels=batch).loss          # shift handled inside the model
        loss.backward(); opt.step(); opt.zero_grad(); tot += loss.item()
    print(f"epoch {ep}/{args.epochs}   loss = {tot/len(dl):.4f}   perplexity = {torch.exp(torch.tensor(tot/len(dl))):.1f}")

banner("AFTER fine-tuning")
results = []
for p in prompts:
    torch.manual_seed(1)
    for g in generate(p, n=2):
        print(f"> {g}\n"); results.append(g)

(OUTPUT_DIR / "26_generated_text.txt").write_text("\n\n".join(results), encoding="utf-8")
model.save_pretrained(OUTPUT_DIR / "26_gpt2_finetuned"); tok.save_pretrained(OUTPUT_DIR / "26_gpt2_finetuned")
print("Generated text saved to outputs/26_generated_text.txt, model saved to outputs/26_gpt2_finetuned")
