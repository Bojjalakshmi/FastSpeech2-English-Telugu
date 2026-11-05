
"""
Utility to inspect audio and create utterance-level mel files if desired.
Token-level duration/pitch/energy sidecars must come from a forced aligner / aligned
feature extraction pipeline; this script deliberately does not invent them.
"""
import argparse, pandas as pd, os, numpy as np
from .audio import load_audio, mel_spectrogram
p=argparse.ArgumentParser()
p.add_argument("--csv", required=True); p.add_argument("--sample_rate", type=int, default=22050)
p.add_argument("--hop_length", type=int, default=256); a=p.parse_args()
df=pd.read_csv(a.csv)
for path in df.audio.astype(str):
    wav=load_audio(path, a.sample_rate)
    mel=mel_spectrogram(wav, sample_rate=a.sample_rate, hop_length=a.hop_length)
    np.save(path+".mel.npy", mel)
print("Mel feature extraction complete. Token alignments are still required.")
