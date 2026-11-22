"""Fair comparison on the v3 test set: MLE, binned MLE 5x5, FLIMngo pretrained, FLIMngo retrained on
v3 (checkpoints/flimngo_v3.pt) and DiffFLIM v2 (3 h and 6 h).

Masks: FLIMngo gets a background mask COMPUTED from the counts (src/evaluation/calibration.py::
foreground_mask). Main mask since 2026-11-22: "flimngo_recall" (chosen on the calibration sets with
scripts/select_flimngo_mask.py); the earlier mask "otsu_adaptive" (precision-oriented, also used to
train the retrained FLIMngo) is kept in the table as a second row per method (column "mask").
DiffFLIM, MLE and binned MLE get no mask at all. All errors are scored on the true foreground
("fg") and on bright foreground pixels ("bright", metrics.bright_mask), as before.

DiffFLIM rows are computed here from cached test predictions (10 samples, seeds 1000 + i, as in
eval_diffflim.py and calibrate_diffflim.py) for 10 DDIM steps (default since 2026-11-22) and 50
steps; missing caches are created. MLE / binned MLE rows are read from results/tables/baselines_v3.csv.

Writes results/tables/compare_v2.csv
       results/figures/error_vs_photons_compare_v2.png

Usage:
    python scripts/eval_compare_v2.py [--ckpt checkpoints/flimngo_v3.pt]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.baselines import flimngo_baseline as pre  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.evaluation.calibration import foreground_mask  # noqa: E402
from src.evaluation.metrics import BRIGHT_FRAC, bright_mask, summarize  # noqa: E402
from src.models.flimngo_arch import build_flimngo, predict as predict_arch  # noqa: E402
from src.paths import CHECKPOINTS, FIGURES, RAW_OWN, TABLES  # noqa: E402

MASKS = ("flimngo_recall", "otsu_adaptive")       # first = main
DIFFFLIM = [("DiffFLIM v2 (3 h)", "diffflim_v2_180min.pt", "v2_180min"), ("DiffFLIM v2 (6 h)", "diffflim_v2.pt", "v2")]
STEPS = (10, 50)                                    # first = default


def load_retrained(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build_flimngo(ck["arch"]["image_size"], ck["arch"]["bin_width"]).to(device)
    net.load_state_dict(ck["model"])
    return net.eval(), ck


@torch.no_grad()
def run_retrained(net, counts, masks, batch=8):
    out = np.zeros((len(counts),) + counts.shape[-2:], np.float32)
    for i in range(0, len(counts), batch):
        c = torch.from_numpy(counts[i:i + batch].astype(np.float32)).cuda()
        m = torch.from_numpy(masks[i:i + batch]).cuda()
        out[i:i + batch] = predict_arch(net, c, m).float().cpu().numpy()
    return out


def time_retrained(net, counts, mask, reps=20):
    c = torch.from_numpy(counts[None].astype(np.float32)).cuda()
    m = torch.from_numpy(mask[None]).cuda()
    with torch.no_grad():
        predict_arch(net, c, m)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            predict_arch(net, c, m)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def diffflim_preds(ckpt, ptag, steps, test_files):
    """Cached DiffFLIM test predictions (all images, 10 samples, seeds 1000 + i)."""
    import calibrate_diffflim as cd

    def model_fn():
        from src.models.diffflim import load_model
        return load_model(CHECKPOINTS / ckpt, "cuda")[0]
    return cd.load_or_predict(cd.preds_cache(ptag, "test", steps, 10), model_fn, test_files, 0, 10, 1000, False, steps)


def row(method, mask, steps, p, subset, n, st, si=None, inside=None, recall=None):
    return {"method": method, "mask": mask, "ddim_steps": steps, "photons": p, "subset": subset, "n_images": n,
            "tau_mare_pct": st["mare_pct"], "tau_bias_pct": st["bias_pct"],
            "int_mare_pct": si["mare_pct"] if si else "", "int_bias_pct": si["bias_pct"] if si else "",
            "tau_mare_inside_mask_pct": inside if inside is not None else "",
            "mask_recall": recall if recall is not None else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/flimngo_v3.pt")
    args = ap.parse_args()
    net, ck = load_retrained(ROOT / args.ckpt)
    print(f"Retrained FLIMngo: {args.ckpt}, step {ck['step']}, {ck['elapsed_s'] / 60:.0f} min, "
          f"validation MSE {ck['val_loss']:.4f}")

    rows = []
    files = sorted((RAW_OWN / "sim_v3_test").glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    for path in files:
        d = load_dataset(path)
        p = int(round(float(d["photons"][0])))
        fg = d["mask"]
        subsets = {"fg": fg, "bright": bright_mask(d["intensity_gt"], fg)}
        for mm in MASKS:
            masks = np.stack([foreground_mask(c.sum(0), mm) for c in d["counts"]])
            recall = float((masks & fg).sum() / fg.sum())
            preds = {"FLIMngo pretrained": pre.predict(d["counts"], masks, bin_width=float(d["metadata"]["bin_width_ns"]))[0],
                     "FLIMngo retrained v3": run_retrained(net, d["counts"], masks)}
            for name, tau in preds.items():
                for subset, m in subsets.items():
                    st = summarize(tau, d["tau_gt"], m)
                    inside = summarize(tau, d["tau_gt"], m & masks)["mare_pct"]
                    rows.append(row(name, mm, "", p, subset, len(fg), st, inside=inside, recall=recall))
                    print(f"p={p:5d} {name:21s} {mm:14s} {subset:6s} tau err {st['mare_pct']:6.2f}%  "
                          f"bias {st['bias_pct']:+6.2f}%  (inside mask {inside:.2f}%, recall {recall:.3f})", flush=True)
    t_re = time_retrained(net, d["counts"][0], masks[0])
    print(f"Retrained FLIMngo: {1e3 * t_re:.1f} ms per 64x64 image (batch 1, network only)")

    for label, ckpt, ptag in DIFFFLIM:
        for steps in STEPS:
            rec = diffflim_preds(ckpt, ptag, steps, files)
            for p in sorted(set(np.round(rec["photons"]).astype(int))):
                idx = np.where(np.round(rec["photons"]) == p)[0]
                fg = rec["mask"][idx]
                for subset, m in {"fg": fg, "bright": bright_mask(rec["int_gt"][idx], fg)}.items():
                    st = summarize(rec["tau_mean"][idx], rec["tau_gt"][idx], m)
                    si = summarize(rec["int_mean"][idx], rec["int_gt"][idx], m)
                    rows.append(row(label, "", steps, int(p), subset, len(idx), st, si))
            print(f"{label}, {steps} steps: " + "  ".join(
                f"{r['photons']}: {r['tau_mare_pct']:.2f}%" for r in rows
                if r["method"] == label and r["ddim_steps"] == steps and r["subset"] == "fg"), flush=True)

    for r in _read(TABLES / "baselines_v3.csv"):
        if r["method"] in ("MLE", "BinMLE5"):
            st = {"mare_pct": float(r["tau_mare_pct"]), "bias_pct": float(r["tau_bias_pct"])}
            si = {"mare_pct": float(r["int_mare_pct"]), "bias_pct": float(r["int_bias_pct"])}
            rows.append(row({"MLE": "MLE", "BinMLE5": "Binned MLE 5x5"}[r["method"]], "", "", int(r["photons"]),
                            r["subset"], int(r["n_images"]), st, si))

    out = TABLES / "compare_v2.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out}")
    plot(rows, FIGURES / "error_vs_photons_compare_v2.png")


def plot(rows, out):
    """Main settings only: FLIMngo with the recall mask, DiffFLIM with 10 DDIM steps."""
    sel = lambda r, m: r["method"] == m and r["mask"] in ("", MASKS[0]) and r["ddim_steps"] in ("", STEPS[0])  # noqa: E731
    styles = {"MLE": ("o-", "C0"), "Binned MLE 5x5": ("v-", "C3"), "FLIMngo pretrained": ("P--", "C5"),
              "FLIMngo retrained v3": ("P-", "C1"), "DiffFLIM v2 (3 h)": ("D--", "C9"), "DiffFLIM v2 (6 h)": ("D-", "C4")}
    titles = {"fg": "all foreground pixels", "bright": f"bright fg pixels (>= {BRIGHT_FRAC:.0%} of intensity range)"}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    for i, subset in enumerate(["fg", "bright"]):
        for j, (key, qname) in enumerate([("tau_mare_pct", "Lifetime"), ("int_mare_pct", "Intensity")]):
            a = ax[i, j]
            for m, (ls, c) in styles.items():
                pts = sorted((r["photons"], r[key]) for r in rows if sel(r, m) and r["subset"] == subset and r[key] != "")
                if pts:
                    lab = m + (" (recall mask)" if m.startswith("FLIMngo") else "") + \
                        (" (10 DDIM steps)" if m.startswith("DiffFLIM") else "")
                    a.plot([x for x, _ in pts], [y for _, y in pts], ls, color=c, label=lab)
            a.set_xscale("log"); a.set_yscale("log"); a.grid(True, which="both", alpha=0.3)
            a.set_xlabel("Photons per foreground pixel"); a.set_ylabel("Median |relative error| (%)")
            a.set_title(f"{qname} error, {titles[subset]}", fontsize=10); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
