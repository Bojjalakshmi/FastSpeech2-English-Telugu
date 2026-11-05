
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .text import encode, normalize_text
from .audio import load_audio, mel_spectrogram

class TTSDataset(Dataset):
    def __init__(self, csv_path, vocab, config):
        self.df = pd.read_csv(csv_path)
        if not {"audio", "text"}.issubset(self.df.columns):
            raise ValueError("CSV must contain columns: audio,text")
        self.vocab, self.cfg = vocab, config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = str(row.audio)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(audio_path)
        text = normalize_text(row.text)
        tokens = encode(text, self.vocab)
        wav = load_audio(audio_path, self.cfg["audio"]["sample_rate"])
        mel = mel_spectrogram(
            wav, sample_rate=self.cfg["audio"]["sample_rate"],
            n_fft=self.cfg["audio"]["n_fft"], hop_length=self.cfg["audio"]["hop_length"],
            win_length=self.cfg["audio"]["win_length"], n_mels=self.cfg["audio"]["n_mels"],
            fmin=self.cfg["audio"]["fmin"], fmax=self.cfg["audio"]["fmax"]
        )
        dur_path = audio_path + ".durations.npy"
        pitch_path = audio_path + ".pitch.npy"
        energy_path = audio_path + ".energy.npy"
        if not os.path.isfile(dur_path):
            raise FileNotFoundError(
                f"Missing token-duration alignment: {dur_path}. "
                "FastSpeech 2 training requires one frame-duration per text token. "
                "Create alignments with a suitable forced aligner; do not use utterance duration."
            )
        durations = np.load(dur_path).astype(np.int64)
        if len(durations) != len(tokens):
            raise ValueError(f"{audio_path}: {len(durations)} durations but {len(tokens)} text tokens")
        if durations.sum() <= 0:
            raise ValueError(f"{audio_path}: duration sum must be positive")
        if abs(int(durations.sum()) - len(mel)) > max(5, int(0.15 * len(mel))):
            raise ValueError(f"{audio_path}: durations sum ({durations.sum()}) do not match mel frames ({len(mel)}).")
        if os.path.isfile(pitch_path):
            pitch = np.load(pitch_path).astype(np.float32)
            if len(pitch) != len(tokens):
                raise ValueError("pitch sidecar must have one value per token")
        else:
            pitch = np.zeros(len(tokens), np.float32)
        if os.path.isfile(energy_path):
            energy = np.load(energy_path).astype(np.float32)
            if len(energy) != len(tokens):
                raise ValueError("energy sidecar must have one value per token")
        else:
            energy = np.zeros(len(tokens), np.float32)
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mel": torch.tensor(mel, dtype=torch.float32),
            "durations": torch.tensor(durations, dtype=torch.long),
            "pitch": torch.tensor(pitch, dtype=torch.float32),
            "energy": torch.tensor(energy, dtype=torch.float32)
        }

def collate_fn(batch):
    b = len(batch)
    max_t = max(x["tokens"].numel() for x in batch)
    max_m = max(x["mel"].size(0) for x in batch)
    n_mels = batch[0]["mel"].size(1)
    tokens = torch.zeros(b, max_t, dtype=torch.long)
    durations = torch.zeros(b, max_t, dtype=torch.long)
    pitch = torch.zeros(b, max_t)
    energy = torch.zeros(b, max_t)
    mel = torch.zeros(b, max_m, n_mels)
    token_lens, mel_lens = [], []
    for i, item in enumerate(batch):
        t, m = item["tokens"].numel(), item["mel"].size(0)
        tokens[i, :t] = item["tokens"]
        durations[i, :t] = item["durations"]
        pitch[i, :t] = item["pitch"]
        energy[i, :t] = item["energy"]
        mel[i, :m] = item["mel"]
        token_lens.append(t)
        mel_lens.append(m)
    return tokens, torch.tensor(token_lens), durations, pitch, energy, mel, torch.tensor(mel_lens)
