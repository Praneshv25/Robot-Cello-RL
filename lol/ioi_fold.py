#!/usr/bin/env python3
# ioi_fold_oneplot.py
import argparse, os, sys, math
from importlib.machinery import SourceFileLoader

import numpy as np
import matplotlib.pyplot as plt

def load_compare(lib_path):
    mod = SourceFileLoader("ioi2_mod", lib_path).load_module()
    if not hasattr(mod, "compare"):
        raise AttributeError(f"'compare' not found in {lib_path}")
    return mod.compare

def list_by_stem(root, exts):
    out = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                stem = os.path.splitext(fn)[0]
                out[stem] = os.path.join(dirpath, fn)
    return out

def pick_ratio_series(df):
    candidates = [
        "ratio", "ratios", "ratio_all",
        "ioi_ratio", "ioi_ratios",
        "beat_ratio", "beat_ratios",
    ]
    for c in candidates:
        if c in df.columns:
            vals = df[c].to_numpy(dtype=float, copy=False)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                return vals
    # Last resort: look for any column containing 'ratio'
    for c in df.columns:
        if "ratio" in c.lower():
            vals = df[c].to_numpy(dtype=float, copy=False)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                return vals
    return np.array([], dtype=float)

def coalesce_metric(overall):
    """
    Choose one scalar 'average-like' metric to print for the whole dataset
    if available in overall. We still compute our own global mean below.
    """
    for k in ("score", "ratio_mean", "mean_ratio"):
        v = overall.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    stats = overall.get("ratio_all_stats") or {}
    v = stats.get("mean")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Batch IOI compare for two folders → ONE overall average + ONE graph."
    )
    ap.add_argument("--midi_dir", required=True, help="Folder with .mid/.midi")
    ap.add_argument("--audio_dir", required=True, help="Folder with audio files")
    ap.add_argument("--lib", default="ioi-2.py", help="Path to your ioi-2.py (must define compare())")
    ap.add_argument("--allowed", type=str, default="0.5,1.0,2.0,4.0",
                    help="Comma-separated allowed IOIs in beats (ignored if --auto-allowed).")
    ap.add_argument("--auto-allowed", action="store_true",
                    help="Infer allowed IOIs from MIDI (passed to compare).")
    ap.add_argument("--max-rel-err", type=float, default=0.35)
    ap.add_argument("--ignore-long-beats", type=float, default=None)
    ap.add_argument("--plots", action="store_true",
                    help="If your compare() tries to plot per-pair, this flag will pass True. Default False.")
    ap.add_argument("--midi_exts", default=".mid,.midi")
    ap.add_argument("--audio_exts", default=".wav,.mp3,.flac,.ogg,.m4a")
    ap.add_argument("--save_plot", default=None,
                    help="If set, save the aggregated figure to this path instead of showing it.")
    ap.add_argument("--title", default="Aggregated IOI Ratios (All Pieces)",
                    help="Title for the single aggregated plot.")
    args = ap.parse_args()

    # Load compare()
    compare = load_compare(args.lib)

    midi_exts  = tuple(e.strip().lower() for e in args.midi_exts.split(",") if e.strip())
    audio_exts = tuple(e.strip().lower() for e in args.audio_exts.split(",") if e.strip())

    midi_map  = list_by_stem(args.midi_dir, midi_exts)
    audio_map = list_by_stem(args.audio_dir, audio_exts)

    stems = sorted(set(midi_map).intersection(audio_map))
    if not stems:
        print(f"No matching stems between {args.midi_dir} and {args.audio_dir}.", file=sys.stderr)
        sys.exit(2)

    allowed_vals = tuple(float(x.strip()) for x in args.allowed.split(",") if x.strip())

    all_ratios = []
    per_pair_metrics = []
    total_pairs = 0
    successes = 0
    failures = []

    print(f"Found {len(stems)} matching pairs. Processing...\n")
    for stem in stems:
        midi_path  = midi_map[stem]
        audio_path = audio_map[stem]
        print(f"→ {stem}\n   MIDI : {midi_path}\n   AUDIO: {audio_path}")
        try:
            df, overall = compare(
                midi_path, audio_path,
                allowed_vals=allowed_vals,
                max_rel_err=args.max_rel_err,
                ignore_long_beats=args.ignore_long_beats,
                plots=args.plots,
                auto_allowed=args.auto_allowed
            )
            ratios = pick_ratio_series(df)
            if ratios.size:
                all_ratios.append(ratios)
            m = coalesce_metric(overall)
            if m is not None and math.isfinite(m):
                per_pair_metrics.append(m)

            successes += 1
        except Exception as e:
            print(f"   [ERROR] {e}", file=sys.stderr)
            failures.append((stem, str(e)))
        finally:
            total_pairs += 1

    if not all_ratios:
        print("\nNo ratio data collected from any pair. Nothing to plot.", file=sys.stderr)
        if failures:
            print("Failures:")
            for stem, err in failures:
                print(f" - {stem}: {err}", file=sys.stderr)
        sys.exit(3)

    # Aggregate
    cat = np.concatenate(all_ratios, axis=0)
    cat = cat[np.isfinite(cat)]

    # Overall stats (dataset-wide)
    glob_mean = float(np.mean(cat)) if cat.size else float("nan")
    glob_median = float(np.median(cat)) if cat.size else float("nan")
    glob_std = float(np.std(cat, ddof=1)) if cat.size > 1 else float("nan")
    glob_n = int(cat.size)

    # (Optional) average of per-pair means/scores if available
    per_pair_avg = float(np.mean(per_pair_metrics)) if per_pair_metrics else None

    # Console report — SINGLE OVERALL AVERAGE
    print("\n==== SINGLE DATASET-WIDE AVERAGE COMPARISON ====")
    print(f"Pairs processed: {successes}/{total_pairs}")
    print(f"Total ratio samples: {glob_n}")
    print(f"Global mean ratio: {glob_mean:.6g}")
    print(f"Global median ratio: {glob_median:.6g}")
    if math.isfinite(glob_std):
        print(f"Global std dev: {glob_std:.6g}")
    if per_pair_avg is not None and math.isfinite(per_pair_avg):
        print(f"Mean of per-pair metric (e.g., score/ratio_mean): {per_pair_avg:.6g}")

    # ONE PLOT (histogram of all ratios)
    plt.figure(figsize=(9, 5.5))
    plt.hist(cat, bins=60, alpha=0.85)
    plt.axvline(1.0, linestyle="--", linewidth=2)      # reference: perfect timing
    plt.axvline(glob_mean, linestyle="-", linewidth=2) # dataset mean
    plt.title(args.title)
    plt.xlabel("IOI Ratio (Audio / MIDI in beats)")
    plt.ylabel("Count")
    plt.tight_layout()

    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150)
        print(f"\nPlot saved to: {args.save_plot}")
    else:
        plt.show()

    if failures:
        print("\nPairs that failed:")
        for stem, err in failures:
            print(f" - {stem}: {err}", file=sys.stderr)

if __name__ == "__main__":
    main()
