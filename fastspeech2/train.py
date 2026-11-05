
import argparse, os, yaml, json, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from .data import TTSDataset, collate_fn
from .model import FastSpeech2
from .text import build_vocab, save_vocab

def masked_l1(pred, target, lengths):
    mask = torch.arange(target.size(1), device=target.device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).expand_as(target)
    return F.l1_loss(pred[mask], target[mask])

def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total, count = 0.0, 0
    for batch in tqdm(loader, leave=False):
        tokens, tl, dur, pitch, energy, mel, ml = [x.to(device) for x in batch]
        if train:
            optimizer.zero_grad(set_to_none=True)
        out = model(tokens, tl, dur, pitch, energy)
        mel_loss = masked_l1(out["mel"], mel, ml)
        dur_target = torch.log(dur.float() + 1)
        dur_loss = F.mse_loss(out["log_duration"], dur_target)
        pitch_loss = F.mse_loss(out["pitch"], pitch)
        energy_loss = F.mse_loss(out["energy"], energy)
        loss = mel_loss + dur_loss + pitch_loss + energy_loss
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--valid_csv", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    train_df = __import__("pandas").read_csv(args.train_csv)
    vocab = build_vocab(train_df.text.astype(str).tolist())
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    vocab_path = os.path.join(cfg["paths"]["checkpoint_dir"], "vocab.json")
    save_vocab(vocab, vocab_path)
    train_ds = TTSDataset(args.train_csv, vocab, cfg)
    valid_ds = TTSDataset(args.valid_csv, vocab, cfg)
    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                          shuffle=True, collate_fn=collate_fn, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=cfg["training"]["batch_size"],
                          shuffle=False, collate_fn=collate_fn, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSpeech2(len(vocab), **cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    best = float("inf")
    train_hist, valid_hist = [], []
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        tr = run_epoch(model, train_dl, optimizer, device, True)
        with torch.no_grad():
            va = run_epoch(model, valid_dl, optimizer, device, False)
        train_hist.append(tr); valid_hist.append(va)
        print(f"Epoch {epoch}: train={tr:.4f} valid={va:.4f}")
        state = {"model": model.state_dict(), "vocab_size": len(vocab),
                 "config": cfg, "epoch": epoch, "valid_loss": va}
        torch.save(state, os.path.join(cfg["paths"]["checkpoint_dir"], "latest.pt"))
        if va < best:
            best = va
            torch.save(state, os.path.join(cfg["paths"]["checkpoint_dir"], "best.pt"))
        for name, hist in [("training_loss.png", train_hist), ("validation_loss.png", valid_hist)]:
            plt.figure()
            plt.plot(range(1, len(hist)+1), hist)
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(name.replace("_", " "))
            plt.tight_layout()
            outdir = cfg["paths"]["results_dir"]
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, name)); plt.close()

if __name__ == "__main__":
    main()
