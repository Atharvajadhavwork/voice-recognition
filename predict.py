"""Predict the spoken word in one or more .wav files.

    python predict.py path/to/clip.wav [more.wav ...]
"""
import argparse

import torch

from src.dataset import load_wav
from src.model import KeywordCNN


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = KeywordCNN(len(ckpt["labels"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["labels"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wavs", nargs="+")
    p.add_argument("--ckpt", default="checkpoints/kws_cnn.pt")
    p.add_argument("--top", type=int, default=3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, labels = load_model(args.ckpt, device)
    for path in args.wavs:
        wav = torch.from_numpy(load_wav(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(wav).softmax(1)[0]
        top = probs.topk(args.top)
        guesses = ", ".join(f"{labels[i]} ({p:.1%})" for p, i in zip(top.values, top.indices))
        print(f"{path}: {guesses}")


if __name__ == "__main__":
    main()
