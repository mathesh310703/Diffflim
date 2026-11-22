"""Where does the positive lifetime bias of the retrained FLIMngo (checkpoints/flimngo_v3.pt) come from?

No training. On the fixed v3 test set (240 images), all true foreground pixels:
  * bias (median of (estimate - truth) / truth) and error, split by photon level, true lifetime,
    number of decay components, IRF, and by whether the pixel lies inside the computed mask;
  * for: FLIMngo retrained (best and last checkpoint; training mask "otsu_adaptive", the
    recall mask "flimngo_recall" and the ground-truth mask), FLIMngo pretrained (recall mask and
    ground-truth mask) and DiffFLIM v2 (6 h; cached test predictions, 50 and 10 DDIM steps).
Checks for a mismatch between training and evaluation:
  (1) the training input pipeline (scripts/train_flimngo.py::to_item, int16 counts) and the
      evaluation pipeline (float counts) give the same prediction on test images;
  (2) the training stream and the test set have the same distributions of true lifetime, number
      of components, IRF, IRF shift and background per photon;
  (3) bias on images simulated from TRAINING cells at the test photon levels, and on the
      validation set used during training (training cells, log-uniform photons).

Writes results/tables/flimngo_bias_breakdown.csv, results/tables/flimngo_bias_checks.csv,
       results/figures/flimngo_bias_breakdown.png

Usage:
    python scripts/analyse_flimngo_bias.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_flimngo as tf  # noqa: E402
from eval_compare_v2 import load_retrained, run_retrained  # noqa: E402
from src.baselines import flimngo_baseline as pre  # noqa: E402
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.evaluation.calibration import foreground_mask  # noqa: E402
from src.models.flimngo_arch import predict as predict_arch  # noqa: E402
from src.paths import CHECKPOINTS, FIGURES, PROCESSED, RAW_OWN, TABLES  # noqa: E402

LEVELS = (10, 20, 50, 100, 500, 1000)
TAU_BINS = (0.1, 1.0, 2.0, 3.0, 4.0, 6.0, 12.0)


def load_test():
    files = sorted((RAW_OWN / "sim_v3_test").glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    ds = [load_dataset(f) for f in files]
    keys = ["counts", "tau_gt", "mask", "n_components", "irf_id", "irf_shift", "photons", "bg_per_bin", "intensity_gt"]
    return {k: np.concatenate([d[k] for d in ds]) for k in keys}


def masks_for(counts, method):
    return np.stack([foreground_mask(c.sum(0), method) for c in counts])


def predictions(test):
    """{method label: tau map (N, H, W)} on the test set."""
    out = {}
    m_train = masks_for(test["counts"], "otsu_adaptive")
    m_recall = masks_for(test["counts"], "flimngo_recall")
    m_true = test["mask"]
    best, ck = load_retrained(CHECKPOINTS / "flimngo_v3.pt")
    last, ck_last = load_retrained(CHECKPOINTS / "flimngo_v3_last.pt")
    print(f"best checkpoint: step {ck['step']}, last: step {ck_last['step']}")
    out["retrained best, otsu_adaptive mask"] = run_retrained(best, test["counts"], m_train)
    out["retrained best, recall mask"] = run_retrained(best, test["counts"], m_recall)
    out["retrained best, true mask"] = run_retrained(best, test["counts"], m_true)
    out["retrained last, otsu_adaptive mask"] = run_retrained(last, test["counts"], m_train)
    out["retrained last, recall mask"] = run_retrained(last, test["counts"], m_recall)
    bw = 0.0977
    out["pretrained, recall mask"] = pre.predict(test["counts"], m_recall, bin_width=bw)[0]
    out["pretrained, true mask"] = pre.predict(test["counts"], m_true, bin_width=bw)[0]
    for name, f in [("DiffFLIM v2 6 h, 50 steps", "preds_v2_test.npz"), ("DiffFLIM v2 6 h, 10 steps", "preds_v2_s10_n10_test.npz")]:
        if (PROCESSED / f).exists():
            with np.load(PROCESSED / f) as z:
                assert np.allclose(z["tau_gt"], test["tau_gt"]), f
                out[name] = z["tau_mean"]
    return out, m_train, best


def stats(est, gt):
    rel = (est - gt) / gt
    return {"n_pixels": int(rel.size), "bias_pct": 100 * float(np.median(rel)),
            "mare_pct": 100 * float(np.median(np.abs(rel))), "bias_ns": float(np.median(est - gt)),
            "mean_rel_pct": 100 * float(np.mean(rel))}


def _group_key(g):
    """Numeric order for photon levels, lifetime bins and component numbers; text order otherwise."""
    try:
        return (0, float(g.split("-")[0]), g)
    except ValueError:
        return (1, 0.0, g)


def breakdown(test, preds, m_train):
    fg = test["mask"]
    n = len(fg)
    per_img = lambda a: np.broadcast_to(np.asarray(a).reshape(n, 1, 1), fg.shape)[fg]  # noqa: E731
    gt = test["tau_gt"][fg]
    factors = {
        "all": np.full(gt.shape, "all"),
        "photons": per_img(np.round(test["photons"]).astype(int)).astype(str),
        "tau_gt_ns": np.array([f"{lo:g}-{hi:g}" for lo, hi in zip(TAU_BINS[:-1], TAU_BINS[1:])])[
            np.clip(np.digitize(gt, TAU_BINS) - 1, 0, len(TAU_BINS) - 2)],
        "n_components": test["n_components"][fg].astype(str),
        "irf": per_img(test["irf_id"]),
        "inside_otsu_adaptive_mask": m_train[fg].astype(str),
    }
    rows = []
    for name, tau in preds.items():
        est = tau[fg].astype(np.float64)
        for fac, lab in factors.items():
            for g in sorted(set(lab), key=_group_key):
                sel = lab == g
                rows.append({"method": name, "factor": fac, "group": g, **stats(est[sel], gt[sel])})
    return rows


def check_pipeline(net, test, n=24):
    """Prediction through the training input path (to_item) vs the evaluation path."""
    idx = np.linspace(0, len(test["counts"]) - 1, n).astype(int)
    diffs = []
    for i in idx:
        s = {"counts": test["counts"][i].astype(np.int64), "mask": test["mask"][i], "tau_gt": test["tau_gt"][i]}
        it = tf.to_item(s)
        with torch.no_grad():
            a = predict_arch(net, it["counts"][None].cuda().float(), it["mask"][None].cuda())[0].cpu().numpy()
        b = run_retrained(net, test["counts"][i:i + 1], masks_for(test["counts"][i:i + 1], "otsu_adaptive"))[0]
        diffs.append(float(np.abs(a - b).max()))
    return max(diffs)


def stream_vs_test(test, n=600):
    """Distributions of the training stream (DiffFLIM v2 training IDs) and of the test set."""
    ids = v3.diffflim_v2_train_ids()
    gen = v3.training_samples(None, (10, 1000), seed=4242, ids=ids)
    S = [next(gen) for _ in range(n)]

    def summ(tau, ncomp, irf, shift, bgratio):
        q = np.percentile(tau, [10, 25, 50, 75, 90])
        c = Counter(ncomp.tolist()); tot = sum(c.values())
        r = {f"tau_p{p}": float(v) for p, v in zip([10, 25, 50, 75, 90], q)}
        r.update({f"ncomp_{k}": c.get(k, 0) / tot for k in (1, 2, 3, 4)})
        ic = Counter(irf.tolist())
        r["irf_max_share"] = max(ic.values()) / len(irf); r["irf_min_share"] = min(ic.values()) / len(irf)
        r["n_irfs"] = len(ic)
        r["irf_shift_mean"] = float(np.mean(shift)); r["irf_shift_sd"] = float(np.std(shift))
        r["bg_photons_per_signal_photon"] = float(np.median(bgratio))
        return r

    tr = summ(np.concatenate([s["tau_gt"][s["mask"]] for s in S]),
              np.concatenate([s["n_components"][s["mask"]] for s in S]),
              np.array([s["irf_id"] for s in S]), np.array([s["irf_shift"] for s in S]),
              np.array([256 * s["bg_per_bin"] / s["photons"] for s in S]))
    fg = test["mask"]
    te = summ(test["tau_gt"][fg], test["n_components"][fg], test["irf_id"], test["irf_shift"],
              256 * test["bg_per_bin"] / test["photons"])
    return tr, te


def train_cell_sets():
    """(a) 20 images per test photon level from training cells (seed 888), (b) the validation set."""
    ids = v3.diffflim_v2_train_ids()
    rng = np.random.default_rng(888)
    cfg = v3.SimConfigV3()
    items = {"photons": [], "counts": [], "mask": [], "tau_gt": [], "n_components": []}
    for lev in LEVELS:
        for _ in range(20):
            s = v3.simulate_sample(rng, cfg, float(lev), ids)
            items["photons"].append(lev)
            for k in ("counts", "mask", "tau_gt", "n_components"):
                items[k].append(s[k])
    a = {k: np.stack(v) if k != "photons" else np.array(v) for k, v in items.items()}
    val = tf.make_val_set(ids)
    return a, val


def main():
    test = load_test()
    print(f"Test set: {len(test['counts'])} images")
    preds, m_train, best = predictions(test)
    rows = breakdown(test, preds, m_train)
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "flimngo_bias_breakdown.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out}")
    for fac in ("all", "photons", "tau_gt_ns", "n_components", "inside_otsu_adaptive_mask", "irf"):
        print(f"\n== bias % by {fac} (n pixels of first method)")
        groups = [r["group"] for r in rows if r["method"] == next(iter(preds)) and r["factor"] == fac]
        print(f"{'':38s}" + "".join(f"{g[:10]:>11s}" for g in groups))
        for name in preds:
            vals = {r["group"]: r["bias_pct"] for r in rows if r["method"] == name and r["factor"] == fac}
            print(f"{name[:38]:38s}" + "".join(f"{vals[g]:+11.2f}" for g in groups))
        if fac != "all":
            print(f"{'(pixels)':38s}" + "".join(f"{r['n_pixels']:11d}" for r in rows
                                              if r["method"] == next(iter(preds)) and r["factor"] == fac))

    checks = []
    d = check_pipeline(best, test)
    print(f"\n(1) training vs evaluation input path: max |difference| of the prediction {d:.2e} ns")
    checks.append({"check": "train_vs_eval_pipeline_max_abs_diff_ns", "training": d, "test": ""})

    tr, te = stream_vs_test(test)
    print("\n(2) training stream vs test set")
    for k in tr:
        print(f"  {k:30s} train {tr[k]:8.3f}   test {te[k]:8.3f}")
        checks.append({"check": k, "training": tr[k], "test": te[k]})

    a, val = train_cell_sets()
    for name, method in [("best", "flimngo_v3.pt"), ("last", "flimngo_v3_last.pt")]:
        net, _ = load_retrained(CHECKPOINTS / method)
        tau = run_retrained(net, a["counts"], masks_for(a["counts"], "otsu_adaptive"))
        print(f"\n(3a) retrained {name}, training cells at the test levels (otsu_adaptive mask):")
        for lev in LEVELS:
            idx = a["photons"] == lev
            st = stats(tau[idx][a["mask"][idx]], a["tau_gt"][idx][a["mask"][idx]])
            print(f"  {lev:5d} photons: bias {st['bias_pct']:+6.2f} %  error {st['mare_pct']:5.2f} %")
            checks.append({"check": f"train_cells_{name}_bias_pct_{lev}", "training": st["bias_pct"], "test": ""})
        st = stats(tau[a["mask"]], a["tau_gt"][a["mask"]])
        checks.append({"check": f"train_cells_{name}_bias_pct_all", "training": st["bias_pct"], "test": ""})
        lab = np.clip(np.digitize(a["tau_gt"][a["mask"]], TAU_BINS) - 1, 0, len(TAU_BINS) - 2)
        e, g = tau[a["mask"]], a["tau_gt"][a["mask"]]
        print("  by true lifetime: " + "  ".join(
            f"{TAU_BINS[j]:g}-{TAU_BINS[j + 1]:g}: {stats(e[lab == j], g[lab == j])['bias_pct']:+.1f}%"
            for j in range(len(TAU_BINS) - 1) if np.any(lab == j)))
        with torch.no_grad():
            tv = predict_arch(net, val["counts"].cuda().float(), val["mask"].cuda()).cpu().numpy()
        tm = val["true_mask"].numpy()
        st = stats(tv[tm], val["target"].numpy()[tm])
        print(f"(3b) retrained {name}, validation set (training cells): bias {st['bias_pct']:+.2f} %, "
              f"error {st['mare_pct']:.2f} %")
        checks.append({"check": f"validation_{name}_bias_pct", "training": st["bias_pct"], "test": ""})

    # (4) scene-level uncertainty of the test-set bias: the test set is paired, so image i of every
    # level shows the same scene; resample the 40 scenes (all levels of a scene together)
    rng = np.random.default_rng(7)
    fg = test["mask"]
    scene = np.tile(np.arange(40), len(LEVELS))[:len(fg)]
    assert len(fg) == 40 * len(LEVELS)
    for name in ("retrained best, recall mask", "retrained last, recall mask", "pretrained, recall mask",
                 "DiffFLIM v2 6 h, 10 steps"):
        rel = (preds[name] - test["tau_gt"]) / np.where(fg, test["tau_gt"], 1)
        per_scene = [rel[scene == k][fg[scene == k]] for k in range(40)]
        boot = [100 * np.median(np.concatenate([per_scene[k] for k in rng.integers(0, 40, 40)])) for _ in range(300)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sb = [100 * np.median(p) for p in per_scene]
        print(f"(4) {name}: test bias {100 * np.median(rel[fg]):+.2f} %, 95 % scene-bootstrap CI [{lo:+.2f}, {hi:+.2f}] %, "
              f"per-scene bias from {min(sb):+.1f} to {max(sb):+.1f} % (IQR {np.percentile(sb, 25):+.1f} to "
              f"{np.percentile(sb, 75):+.1f} %)")
        checks.append({"check": f"scene_ci_lo_{name}", "training": "", "test": lo})
        checks.append({"check": f"scene_ci_hi_{name}", "training": "", "test": hi})

    out2 = TABLES / "flimngo_bias_checks.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "training", "test"])
        w.writeheader()
        for r in checks:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out2}")
    plot(rows, list(preds))


def plot(rows, methods):
    show = [m for m in methods if m in ("retrained best, otsu_adaptive mask", "retrained best, recall mask",
                                         "retrained last, otsu_adaptive mask", "pretrained, recall mask",
                                         "DiffFLIM v2 6 h, 10 steps", "DiffFLIM v2 6 h, 50 steps")]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for a, fac, title in zip(ax.ravel(), ["photons", "tau_gt_ns", "n_components", "irf"],
                             ["Photon level", "True lifetime (ns)", "Number of components", "IRF"]):
        groups = [r["group"] for r in rows if r["method"] == show[0] and r["factor"] == fac]
        x = np.arange(len(groups))
        wdt = 0.8 / len(show)
        for k, m in enumerate(show):
            v = [next(r["bias_pct"] for r in rows if r["method"] == m and r["factor"] == fac and r["group"] == g)
                 for g in groups]
            a.bar(x + (k - (len(show) - 1) / 2) * wdt, v, wdt, label=m)
        a.axhline(0, color="k", lw=0.8)
        a.set_xticks(x); a.set_xticklabels([g.replace("IRF_", "") for g in groups], rotation=35 if fac == "irf" else 0,
                                           ha="right" if fac == "irf" else "center", fontsize=8)
        a.set_ylabel("Bias: median relative error (%)"); a.set_title(title, fontsize=10); a.grid(alpha=0.3, axis="y")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Lifetime bias on the v3 test set (true foreground pixels)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "flimngo_bias_breakdown.png", dpi=110)
    print(f"Saved {FIGURES / 'flimngo_bias_breakdown.png'}")


if __name__ == "__main__":
    main()
