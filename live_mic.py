"""Real-time keyword spotting from the microphone.

Keeps a rolling 1-second buffer, runs the model every 250 ms and prints a
word when the model is confident and the audio is loud enough to be speech.
"""
import argparse
import queue

import numpy as np
import sounddevice as sd
import torch

from predict import load_model
from src.dataset import SAMPLE_RATE, CLIP_LEN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/kws_cnn.pt")
    p.add_argument("--threshold", type=float, default=0.7, help="min confidence to report")
    p.add_argument("--min-rms", type=float, default=0.01, help="ignore near-silent audio")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, labels = load_model(args.ckpt, device)

    hop = SAMPLE_RATE // 4
    buffer = np.zeros(CLIP_LEN, dtype=np.float32)
    chunks = queue.Queue()
    last = None

    def callback(indata, frames, time, status):
        chunks.put(indata[:, 0].copy())

    print("Listening... say one of:", ", ".join(labels))
    print("Ctrl+C to stop.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=hop, callback=callback):
        try:
            while True:
                chunk = chunks.get()
                buffer = np.concatenate([buffer[len(chunk):], chunk])
                if np.sqrt(np.mean(buffer ** 2)) < args.min_rms:
                    last = None
                    continue
                with torch.no_grad():
                    probs = model(torch.from_numpy(buffer).unsqueeze(0).to(device)).softmax(1)[0]
                conf, idx = probs.max(0)
                word = labels[idx]
                if conf >= args.threshold and word != last:
                    print(f"  heard: {word:10s} ({conf:.0%})")
                    last = word
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
