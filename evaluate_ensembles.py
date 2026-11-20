"""Evaluate every model and every ensemble combination of every annotation step on the test set.

Run from the project folder (the one that contains the annotation_step* folders and test_set/).
Outputs:
  test_scores_step{n}.csv   - test-set scores of each model of step n (for thresholds/curves later)
  ensemble_metrics.csv      - AUC and average precision with 95% bootstrap intervals,
                              for every model and combination of every step
"""
import gc
import itertools
from glob import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# --- settings -----------------------------------------------------------------
TEST_LABELS = ["test_set/test_set_labels_ag.csv", "test_set/test_set_labels_ot.csv"]  # airgun + other part
N_BOOT = 1000
SEED = 3101992
BATCH = 64
# model name: (subfolder inside annotation_step{n}, kind)
MODELS = {
    "resnet": ("resnet/best.model", "cnn"),
    "effnet": ("effnet/best.model", "cnn"),
    "perch":  ("shallow_perch/shallow_classifier.joblib", "perch"),
    "hubert": ("shallow_HUbert/shallow_waveform_classifier.joblib", "hubert"),
}
# -------------------------------------------------------------------------------


def find_case_insensitive(base, rel):
    """Return the path base/rel even if upper/lower case differs (e.g. Shallow_classifier.joblib)."""
    path = Path(base)
    for part in Path(rel).parts:
        if not path.is_dir():
            return None
        match = [p for p in path.iterdir() if p.name.lower() == part.lower()]
        if not match:
            return None
        path = match[0]
    return path


def file_bootstrap(files, n_boot, seed):
    """Resample whole files (both 5 s clips of a 10 s file stay together)."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(files))).groupby(np.asarray(files)).indices   # file -> its row numbers
    uniq = list(groups)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        yield np.concatenate([groups[uniq[i]] for i in pick])


def metrics_with_ci(y, s, boot_sets):
    auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
    aucs, aps = [], []
    for idx in boot_sets:
        yb = y[idx]
        if yb.min() == yb.max():          # resample without both classes: skip
            continue
        aucs.append(roc_auc_score(yb, s[idx]))
        aps.append(average_precision_score(yb, s[idx]))
    lo_auc, hi_auc = np.percentile(aucs, [2.5, 97.5])
    lo_ap, hi_ap = np.percentile(aps, [2.5, 97.5])
    return auc, lo_auc, hi_auc, ap, lo_ap, hi_ap


def evaluate_scores(scores, y, files, n_boot=N_BOOT, seed=SEED):
    """scores: DataFrame, one column per model (probabilities), rows aligned with y."""
    boot_sets = list(file_bootstrap(files, n_boot, seed))   # same resamples for every combination
    rows = []
    cols = list(scores.columns)
    for r in range(1, len(cols) + 1):
        for combo in itertools.combinations(cols, r):
            s = scores[list(combo)].mean(axis=1).to_numpy()
            auc, lo_auc, hi_auc, ap, lo_ap, hi_ap = metrics_with_ci(y, s, boot_sets)
            rows.append({"models": " + ".join(combo), "n_models": r,
                         "auc": auc, "auc_lo": lo_auc, "auc_hi": hi_auc,
                         "ap": ap, "ap_lo": lo_ap, "ap_hi": hi_ap})
    return pd.DataFrame(rows)


def main():
    import opensoundscape as opso
    import torch
    import torchaudio
    import bioacoustics_model_zoo as bmz
    from opensoundscape import Audio

    test_df = pd.concat([pd.read_csv(f, index_col=[0, 1, 2]) for f in TEST_LABELS])
    assert not test_df.index.duplicated().any(), "the same clip appears in both test files"
    test_df = test_df[["airgun"]].astype(int)
    clip_len = float((test_df.index.get_level_values(2) - test_df.index.get_level_values(1)).max())
    y = test_df["airgun"].to_numpy()
    files = test_df.index.get_level_values(0).to_numpy()
    print(f"test set: {len(y)} clips, {y.sum()} airgun, {len(np.unique(files))} files")

    # embeddings are the same for every step, so compute them once
    if Path("test_set/perch_emb.npy").exists():
        perch_emb = np.load("test_set/perch_emb.npy")
    else:
        perch = bmz.Perch2ONNX(headless=True)
        perch_emb = np.asarray(perch.embed(test_df, return_dfs=False, return_preds=False))
        np.save("test_set/perch_emb.npy", perch_emb)
        del perch
    assert len(perch_emb) == len(test_df)

    def hubert_embed_batch(clip_index, batch_size=32):
        bundle = torchaudio.pipelines.HUBERT_BASE          # loaded only when needed
        hubert = bundle.get_model().eval().to("cuda")
        keys, out = list(clip_index), []
        for b in range(0, len(keys), batch_size):
            waves = []
            for f, s, e in keys[b:b + batch_size]:
                x = Audio.from_file(f, offset=s, duration=e - s).resample(bundle.sample_rate).samples
                n = int(round((e - s) * bundle.sample_rate))
                waves.append(np.pad(x, (0, max(0, n - len(x))))[:n])
            batch = torch.tensor(np.stack(waves), dtype=torch.float32, device="cuda")
            with torch.no_grad():
                feats, _ = hubert.extract_features(batch)
            out.append(feats[-1].mean(dim=1).cpu().numpy())
        del hubert
        torch.cuda.empty_cache()
        return np.concatenate(out)

    hubert_emb = None   # computed on first use

    all_results = []
    steps = sorted(glob("annotation_step*"), key=lambda p: int(p.replace("annotation_step", "")))
    for step_dir in steps:
        step = int(step_dir.replace("annotation_step", ""))
        step_csv = Path(f"ensemble_metrics_step{step}.csv")
        if step_csv.exists():
            print(f"step {step}: already done ({step_csv}), loading")
            all_results.append(pd.read_csv(step_csv))
            continue
        scores = pd.DataFrame(index=test_df.index)
        for name, (rel, kind) in MODELS.items():
            path = find_case_insensitive(step_dir, rel)
            if path is None:
                print(f"step {step}: no {name} ({Path(step_dir) / rel}), skipped")
                continue
            if kind == "cnn":
                m = opso.load_model(path)
                if abs(m.preprocessor.sample_duration - clip_len) > 1e-6:
                    print(f"WARNING step {step} {name}: model uses {m.preprocessor.sample_duration} s clips, "
                          f"test clips are {clip_len} s")
                pred = m.predict(test_df, activation_layer="sigmoid", batch_size=BATCH)
                scores[name] = pred["airgun"].to_numpy()
                del m, pred
                gc.collect()
                torch.cuda.empty_cache()
            elif kind == "perch":
                scores[name] = joblib.load(path).predict_proba(perch_emb)[:, 1]
            elif kind == "hubert":
                if hubert_emb is None:
                    if Path("test_set/hubert_emb.npy").exists():
                        hubert_emb = np.load("test_set/hubert_emb.npy")
                    else:
                        hubert_emb = hubert_embed_batch(test_df.index)
                        np.save("test_set/hubert_emb.npy", hubert_emb)
                scores[name] = joblib.load(path).predict_proba(hubert_emb)[:, 1]
        if scores.empty:
            continue
        scores.assign(airgun=y).to_csv(f"test_scores_step{step}.csv")
        res = evaluate_scores(scores, y, files)
        res.insert(0, "step", step)
        res.to_csv(step_csv, index=False)          # saved right away, so a crash later loses nothing
        all_results.append(res)
        print(res.sort_values("auc", ascending=False).head(5).to_string(index=False))

    out = pd.concat(all_results, ignore_index=True)
    out.to_csv("ensemble_metrics.csv", index=False)
    print("\nsaved ensemble_metrics.csv")


if __name__ == "__main__":
    main()