import torch
from torch import nn
import torch.nn.functional as F

class ConvPredictor(nn.Module):
    def __init__(self, hidden, channels=256, kernel=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, channels, kernel, padding=kernel//2),
            nn.ReLU(), nn.GroupNorm(1, channels), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=kernel//2),
            nn.ReLU(), nn.GroupNorm(1, channels), nn.Dropout(dropout),
            nn.Conv1d(channels, 1, 1)
        )
    def forward(self, x):
        # B,T,C -> B,C,T -> B,T
        return self.net(x.transpose(1, 2)).squeeze(1)

class VarianceAdaptor(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.duration_predictor = ConvPredictor(hidden, dropout=dropout)
        self.pitch_predictor = ConvPredictor(hidden, dropout=dropout)
        self.energy_predictor = ConvPredictor(hidden, dropout=dropout)
        self.pitch_embed = nn.Conv1d(1, hidden, 1)
        self.energy_embed = nn.Conv1d(1, hidden, 1)

    def forward(self, x, src_mask, durations=None, pitch=None, energy=None,
                alpha=1.0, pitch_control=1.0, energy_control=1.0):
        log_duration = self.duration_predictor(x).clamp(-5, 8)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)
        p = pitch if pitch is not None else pitch_pred
        e = energy if energy is not None else energy_pred
        x = x + self.pitch_embed((p * pitch_control).unsqueeze(1)).transpose(1, 2)
        x = x + self.energy_embed((e * energy_control).unsqueeze(1)).transpose(1, 2)

        if durations is None:
            durations = torch.clamp(torch.round(torch.exp(log_duration) - 1) * alpha,
                                    min=0, max=100).long()
            durations = durations.masked_fill(src_mask, 0)
        expanded, mel_lens = length_regulate(x, durations)
        return expanded, log_duration, pitch_pred, energy_pred, durations, mel_lens

def length_regulate(x, durations):
    expanded = []
    lengths = []
    for b in range(x.size(0)):
        reps = durations[b].clamp(min=0).tolist()
        seq = torch.repeat_interleave(x[b], torch.tensor(reps, device=x.device), dim=0)
        if seq.size(0) == 0:
            seq = x.new_zeros((1, x.size(-1)))
        expanded.append(seq)
        lengths.append(seq.size(0))
    max_len = max(lengths)
    out = x.new_zeros((x.size(0), max_len, x.size(-1)))
    for b, seq in enumerate(expanded):
        out[b, :seq.size(0)] = seq
    return out, torch.tensor(lengths, device=x.device)

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, hidden=256, n_mels=80, n_heads=2,
                 n_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4,
            dropout=dropout, batch_first=True, activation="relu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.variance_adaptor = VarianceAdaptor(hidden, dropout)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4,
            dropout=dropout, batch_first=True, activation="relu"
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_layers)
        self.mel_projection = nn.Linear(hidden, n_mels)

    def forward(self, tokens, token_lens, durations=None, pitch=None, energy=None):
        positions = torch.arange(tokens.size(1), device=tokens.device)[None, :]
        src_mask = positions >= token_lens[:, None]
        x = self.embedding(tokens)
        x = self.encoder(x, src_key_padding_mask=src_mask)
        expanded, log_dur, pitch_pred, energy_pred, used_dur, mel_lens = self.variance_adaptor(
            x, src_mask, durations, pitch, energy
        )
        mel_mask = torch.arange(expanded.size(1), device=tokens.device)[None, :] >= mel_lens[:, None]
        dec = self.decoder(expanded, src_key_padding_mask=mel_mask)
        mel = self.mel_projection(dec)
        return {
            "mel": mel, "log_duration": log_dur, "pitch": pitch_pred,
            "energy": energy_pred, "durations": used_dur,
            "mel_lens": mel_lens, "src_mask": src_mask, "mel_mask": mel_mask
        }
