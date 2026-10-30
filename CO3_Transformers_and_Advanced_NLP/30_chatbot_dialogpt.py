"""
Program 30 : Chatbot Development using a pre-trained transformer (DialoGPT)  [CO3]
Aim        : A conversational agent built on microsoft/DialoGPT (GPT-2 trained on 147M Reddit conversations).
Algorithm  : 1) encode user message + <eos>  2) append to the chat history  3) model.generate continues the dialogue
             4) decode only the NEW tokens  5) keep the last few turns as context (memory)
Install    : pip install torch transformers
Usage      : python 30_chatbot_dialogpt.py                 -> scripted demo (non-interactive)
             python 30_chatbot_dialogpt.py --interactive   -> chat in the terminal, type 'quit' to stop
             python 30_chatbot_dialogpt.py --model microsoft/DialoGPT-medium
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import banner

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="microsoft/DialoGPT-small")
ap.add_argument("--interactive", action="store_true")
ap.add_argument("--max_turns", type=int, default=4, help="how many previous turns to remember")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).eval()
torch.manual_seed(7)

history = []                                      # list of tensors, one per utterance (user and bot)

def reply(user_text):
    global history
    user_ids = tok.encode(user_text + tok.eos_token, return_tensors="pt")
    history.append(user_ids)
    history = history[-(2 * args.max_turns + 1):]                 # sliding window memory
    context = torch.cat(history, dim=-1)
    with torch.no_grad():
        out = model.generate(context, attention_mask=torch.ones_like(context), max_new_tokens=40,
                             do_sample=True, top_k=50, top_p=0.92, temperature=0.7,
                             no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id)
    new_ids = out[:, context.shape[-1]:]                          # only the bot's reply
    history.append(new_ids)
    return tok.decode(new_ids[0], skip_special_tokens=True)

banner(f"Chatbot ready ({args.model})")
if args.interactive:
    print("Type your message. Type 'quit' to exit.\n")
    while True:
        msg = input("You : ").strip()
        if msg.lower() in ("quit", "exit", "bye", ""):
            print("Bot : Goodbye!"); break
        print("Bot :", reply(msg))
else:
    demo = ["Hello, how are you?", "What do you like to do?", "Do you know about natural language processing?",
            "That sounds interesting, tell me more.", "Thanks, bye!"]
    for m in demo:
        print("You :", m)
        print("Bot :", reply(m), "\n")
    print("Run with --interactive to chat yourself.")
