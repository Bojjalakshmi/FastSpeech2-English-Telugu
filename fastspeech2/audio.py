
import numpy as np
import librosa
import soundfile as sf

def load_audio(path, sample_rate=22050):
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != sample_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return wav

def mel_spectrogram(wav, sample_rate=22050, n_fft=1024, hop_length=256,
                    win_length=1024, n_mels=80, fmin=0, fmax=8000):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax,
        power=1.0
    )
    return np.log(np.maximum(mel, 1e-5)).T.astype(np.float32)

def save_wav(path, wav, sample_rate=22050):
    sf.write(path, wav, sample_rate)
