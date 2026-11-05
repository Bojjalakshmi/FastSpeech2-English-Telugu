
import argparse, os, yaml, torch, numpy as np
from .model import FastSpeech2
from .text import load_vocab, encode, normalize_text
from .audio import save_wav

def plot_vector(values, path, title, ylabel):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(max(7, len(values)*0.25), 3))
    plt.plot(np.arange(len(values)), values)
    plt.title(title); plt.xlabel("Text token index"); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(path); plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vocoder", default="")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = load_vocab(os.path.join(os.path.dirname(args.checkpoint), "vocab.json"))
    model = FastSpeech2(len(vocab), **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    tokens = torch.tensor([encode(normalize_text(args.text), vocab)], dtype=torch.long, device=device)
    lens = torch.tensor([tokens.size(1)], device=device)
    with torch.no_grad():
        out = model(tokens, lens)
    mel = out["mel"][0].cpu().numpy()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Required plots: values are token-level predictions from the variance adaptor.
    root = cfg["paths"]["results_dir"]
    tag = os.path.splitext(os.path.basename(args.output))[0]
    plot_vector(out["durations"][0].cpu().numpy(), f"{root}/variance_adaptor_plots/duration/{tag}.png", "Predicted duration", "Frames")
    plot_vector(out["pitch"][0].cpu().numpy(), f"{root}/variance_adaptor_plots/pitch/{tag}.png", "Predicted pitch", "Model units")
    plot_vector(out["energy"][0].cpu().numpy(), f"{root}/variance_adaptor_plots/energy/{tag}.png", "Predicted energy", "Model units")
    if not args.vocoder:
        np.save(args.output + ".mel.npy", mel)
        raise SystemExit(f"Mel saved to {args.output}.mel.npy. Supply --vocoder with a compatible TorchScript vocoder to create WAV.")
    vocoder = torch.jit.load(args.vocoder, map_location=device).eval()
    mel_t = torch.tensor(mel.T[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        wav = vocoder(mel_t).squeeze().cpu().numpy()
    save_wav(args.output, wav, cfg["audio"]["sample_rate"])

if __name__ == "__main__":
    main()
