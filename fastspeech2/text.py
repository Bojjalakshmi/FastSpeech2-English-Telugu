
import re
import unicodedata

PAD = "<pad>"
UNK = "<unk>"

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_vocab(texts):
    chars = sorted(set("".join(normalize_text(t) for t in texts)))
    symbols = [PAD, UNK] + chars
    return {s: i for i, s in enumerate(symbols)}

def encode(text, vocab):
    text = normalize_text(text)
    unk = vocab[UNK]
    return [vocab.get(ch, unk) for ch in text]

def save_vocab(vocab, path):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_vocab(path):
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)
