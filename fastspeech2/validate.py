
import argparse, pandas as pd, os
p=argparse.ArgumentParser(); p.add_argument("--csv", required=True); a=p.parse_args()
df=pd.read_csv(a.csv)
assert {"audio","text"}.issubset(df.columns), "CSV needs audio,text columns"
missing=[x for x in df.audio.astype(str) if not os.path.isfile(x)]
if missing: raise SystemExit(f"{len(missing)} audio paths missing. First: {missing[0]}")
print(f"OK: {len(df)} rows; {df.audio.nunique()} unique audio paths")
