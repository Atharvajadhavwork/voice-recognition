"""Evaluate a trained checkpoint on the test split and plot a confusion matrix."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import SpeechCommands
from src.model import KeywordCNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/speech_commands")
    p.add_argument("--ckpt", default="checkpoints/kws_cnn.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    labels = ckpt["labels"]
    model = KeywordCNN(len(labels)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SpeechCommands(args.data, "test")
    dl = DataLoader(ds, 256, num_workers=8)
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    with torch.no_grad():
        for wav, y in dl:
            pred = model(wav.to(device)).argmax(1).cpu()
            for t, p_ in zip(y, pred):
                cm[t, p_] += 1

    acc = np.trace(cm) / cm.sum()
    per_class = {labels[i]: round(cm[i, i] / cm[i].sum(), 4) for i in range(n)}
    print(f"test accuracy: {acc:.4f}  ({cm.sum()} clips)")
    for name, a in sorted(per_class.items(), key=lambda kv: kv[1])[:5]:
        print(f"  hardest: {name:10s} {a:.3f}")

    with open("results/test_metrics.json", "w") as f:
        json.dump({"test_accuracy": round(acc, 4), "per_class_accuracy": per_class}, f, indent=2)

    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Speech Commands v2 test set — accuracy {acc:.2%}")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=120)
    print("saved results/confusion_matrix.png")


if __name__ == "__main__":
    main()
