"""Log-mel spectrogram front end + a small CNN keyword-spotting classifier."""
import torch
import torch.nn as nn
import torchaudio

from .dataset import SAMPLE_RATE


class LogMel(nn.Module):
    """Raw waveform (B, 16000) -> log-mel spectrogram (B, 1, n_mels, frames)."""

    def __init__(self, n_mels=40):
        super().__init__()
        # 25 ms window, 10 ms hop -> 101 frames per second
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_mels=n_mels)

    def forward(self, wav):
        return torch.log(self.mel(wav) + 1e-6).unsqueeze(1)


class SpecAugment(nn.Module):
    """Randomly mask frequency bands and time steps (training only)."""

    def __init__(self, freq_mask=8, time_mask=20):
        super().__init__()
        self.freq = torchaudio.transforms.FrequencyMasking(freq_mask, iid_masks=True)
        self.time = torchaudio.transforms.TimeMasking(time_mask, iid_masks=True)

    def forward(self, spec):
        if self.training:
            spec = self.time(self.freq(spec))
        return spec


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class KeywordCNN(nn.Module):
    def __init__(self, n_classes, n_mels=40):
        super().__init__()
        self.features = LogMel(n_mels)
        self.norm = nn.InstanceNorm2d(1)  # per-clip normalisation: robust to mic volume
        self.specaug = SpecAugment()
        self.net = nn.Sequential(
            conv_block(1, 32), nn.MaxPool2d(2),
            conv_block(32, 64), nn.MaxPool2d(2),
            conv_block(64, 128), nn.MaxPool2d(2),
            conv_block(128, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, wav):
        x = self.specaug(self.norm(self.features(wav)))
        return self.net(x)
