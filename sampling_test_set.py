"""Random sample from the AMAR sound catalog with airguns at 10% of the other files.

Excluded folders: "Airgun", "Background noise", "test_set".
Writes a CSV with one row per selected file: path, folder, group (airgun / other).
"""
import math
import random
from pathlib import Path
from Keys import root_folder as CATALOG
import pandas as pd

# --- settings -----------------------------------------------------------------
EXCLUDE = {"Airgun", "Background noise", "test_set"}
AIRGUN_DIRS = {"Airgun long", "Airgun or Unknown Sound", "Airguns found", "Airguns irregular"}
AIRGUN_RATIO = 0.10        # number of airgun files = 10% of the number of other files
N_OTHER = 2000             # how many non-airgun files to take; None = all of them
SEED = 3101992
OUT_CSV = Path("catalog_sample.csv")
# -------------------------------------------------------------------------------


def wav_files(folder):
    """All .wav files in a folder and its subfolders (upper or lower case extension)."""
    return sorted(str(p) for p in folder.rglob("*") if p.suffix.lower() == ".wav")


folders = sorted(d for d in CATALOG.iterdir() if d.is_dir() and d.name not in EXCLUDE)
missing = AIRGUN_DIRS - {d.name for d in folders}
if missing:
    print(f"warning: airgun folders not found: {sorted(missing)}")

rows = []
for d in folders:
    group = "airgun" if d.name in AIRGUN_DIRS else "other"
    files = wav_files(d)
    print(f"{d.name:<28} {group:<7} {len(files):>6} files")
    rows += [{"path": f, "folder": d.name, "group": group} for f in files]
pool = pd.DataFrame(rows)

other = pool[pool["group"] == "other"]
airgun = pool[pool["group"] == "airgun"]

n_other = len(other) if N_OTHER is None else min(N_OTHER, len(other))
n_airgun = math.floor(n_other * AIRGUN_RATIO)
if n_airgun > len(airgun):
    # not enough airguns for the ratio: keep the ratio by taking fewer other files
    n_airgun = len(airgun)
    n_other = math.floor(n_airgun / AIRGUN_RATIO)
    print(f"only {len(airgun)} airgun files available, reducing other files to {n_other}")

random.seed(SEED)
sample = pd.concat([
    other.loc[random.sample(list(other.index), n_other)],
    airgun.loc[random.sample(list(airgun.index), n_airgun)],
]).reset_index(drop=True)

sample.to_csv(OUT_CSV, index=False)
print(f"\nselected {n_other} other + {n_airgun} airgun files "
      f"({n_airgun / max(n_other, 1):.0%} airguns relative to other) -> {OUT_CSV}")
print(sample.groupby(["group", "folder"]).size().to_string())