"""Plot training loss and validation accuracy from results/train_history.json."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = json.load(open("results/train_history.json"))
epochs = [e["epoch"] for e in h]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
a.plot(epochs, [e["train_loss"] for e in h], marker="o")
a.set_title("training loss")
a.set_xlabel("epoch")
b.plot(epochs, [100 * e["val_acc"] for e in h], marker="o", color="tab:green")
b.set_title("validation accuracy (%)")
b.set_xlabel("epoch")
for ax in (a, b):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/training_curves.png", dpi=120)
print("saved results/training_curves.png")
