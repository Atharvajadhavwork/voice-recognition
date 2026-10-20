"""Google Speech Commands v2 dataset loader.

Every clip is 1 second of 16 kHz mono audio (shorter clips are zero-padded).
Train/val/test splits come from the official validation_list.txt and
testing_list.txt files, so speakers never overlap between splits.
"""
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
CLIP_LEN = SAMPLE_RATE  # 1 second


def load_wav(path, length=CLIP_LEN):
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE, f"{path} has sample rate {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < length:
        audio = np.pad(audio, (0, length - len(audio)))
    return audio[:length]


def get_labels(root):
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))


class SpeechCommands(Dataset):
    def __init__(self, root, split="train", augment=False):
        self.root = Path(root)
        self.labels = get_labels(self.root)
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.augment = augment

        val = set(self.root.joinpath("validation_list.txt").read_text().split())
        test = set(self.root.joinpath("testing_list.txt").read_text().split())

        self.files = []
        for label in self.labels:
            for wav in sorted((self.root / label).glob("*.wav")):
                rel = f"{label}/{wav.name}"
                in_split = {"val": rel in val, "test": rel in test,
                            "train": rel not in val and rel not in test}[split]
                if in_split:
                    self.files.append((rel, self.label_to_idx[label]))

        # Background noise (white/pink noise, running tap, ...) used for augmentation
        self.noises = []
        if augment:
            for f in sorted((self.root / "_background_noise_").glob("*.wav")):
                self.noises.append(sf.read(f, dtype="float32")[0])

    def __len__(self):
        return len(self.files)

    def _augment(self, audio):
        # Random time shift of up to +-100 ms
        shift = random.randint(-1600, 1600)
        audio = np.roll(audio, shift)
        if shift > 0:
            audio[:shift] = 0
        elif shift < 0:
            audio[shift:] = 0
        # Mix in background noise 80% of the time
        if self.noises and random.random() < 0.8:
            noise = random.choice(self.noises)
            start = random.randint(0, len(noise) - CLIP_LEN)
            audio = audio + random.uniform(0, 0.1) * noise[start:start + CLIP_LEN]
        return audio

    def __getitem__(self, i):
        rel, label = self.files[i]
        audio = load_wav(self.root / rel)
        if self.augment:
            audio = self._augment(audio)
        return torch.from_numpy(audio.astype(np.float32)), label
