"""Train the keyword-spotting CNN on Speech Commands v2 (35 words)."""
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SpeechCommands
from src.model import KeywordCNN


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for wav, y in loader:
            wav, y = wav.to(device), y.to(device)
            correct += (model(wav).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/speech_commands")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="checkpoints/kws_cnn.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = SpeechCommands(args.data, "train", augment=True)
    val_ds = SpeechCommands(args.data, "val")
    print(f"train={len(train_ds)}  val={len(val_ds)}  classes={len(train_ds.labels)}  device={device}")

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.workers, pin_memory=True)

    model = KeywordCNN(len(train_ds.labels)).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for wav, y in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = loss_fn(model(wav), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()

        val_acc = evaluate(model, val_dl, device)
        train_loss = running / len(train_dl)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  val_acc {val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "labels": train_ds.labels}, args.out)

    with open("results/train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val_acc {best:.4f}  -> saved {args.out}")


if __name__ == "__main__":
    main()
