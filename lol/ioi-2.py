import argparse, numpy as np, pandas as pd
import pretty_midi as pm
import librosa
from scipy.signal import medfilt

def choose_instrument(m: pm.PrettyMIDI):
    insts = [i for i in m.instruments if not i.is_drum] or m.instruments
    return max(insts, key=lambda i: np.mean([n.pitch for n in i.notes]) if i.notes else -1)

def extract_midi_onsets_beats(m: pm.PrettyMIDI, dedup_ms=20.0):
    inst = choose_instrument(m)
    if not inst.notes:
        return np.array([]), inst
    notes = sorted(inst.notes, key=lambda n: (n.start, -n.pitch))
    on_s = []
    last = None
    tol = dedup_ms / 1000.0
    for n in notes:
        if last is None or abs(n.start - last) > tol:
            on_s.append(n.start)
            last = n.start
    on_s = np.asarray(on_s, float)
    beat_times = m.get_beats()
    if len(beat_times) < 2:
        on_b = on_s.copy()
    else:
        bidx = np.arange(len(beat_times), dtype=float)
        on_b = np.interp(on_s, beat_times, bidx)
    return on_b, inst

def compute_ioi(x):
    return np.diff(x) if len(x) >= 2 else np.array([])

def infer_allowed_from_midi(m: pm.PrettyMIDI, max_rel_err=0.25, min_count=3,
                            candidate_vals=(0.25,0.5,1.0,2.0,4.0)):
    midi_on_beats, _ = extract_midi_onsets_beats(m)
    ioi_b = np.diff(midi_on_beats)
    if len(ioi_b) == 0:
        return []
    cand = np.asarray(candidate_vals, float)
    nearest = np.argmin(np.abs(ioi_b[:, None] - cand[None, :]), axis=1)
    rel_err = np.abs(ioi_b - cand[nearest]) / (cand[nearest] + 1e-12)
    mask_ok = rel_err <= max_rel_err
    chosen = []
    for j, v in enumerate(cand):
        c = int(np.sum(mask_ok & (nearest == j)))
        if c >= min_count:
            chosen.append(v)
    return sorted(chosen)

def quantize_expected(ioi_beats, allowed=(0.5,1.0,2.0,4.0), max_rel_err=0.35):
    allowed = np.asarray(allowed, float)
    exp = np.empty_like(ioi_beats)
    ok = np.zeros_like(ioi_beats, dtype=bool)
    for i, val in enumerate(ioi_beats):
        j = np.argmin(np.abs(allowed - val))
        exp[i] = allowed[j]
        ok[i] = (abs(val - allowed[j]) / (allowed[j] + 1e-12)) <= max_rel_err
    return exp, ok

def load_audio_onsets_beats(path, sr=22050, hop=256):
    y, sr = librosa.load(path, sr=sr, mono=True)
    y = y/(np.max(np.abs(y))+1e-9)
    on_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True, units="frames")
    on_times = librosa.frames_to_time(on_frames, sr=sr, hop_length=hop)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_env = medfilt(onset_env, kernel_size=5)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    if len(beat_times) < 2:
        on_beats = on_times.copy()
        tempo = 60.0
    else:
        bidx = np.arange(len(beat_times), dtype=float)
        on_beats = np.interp(on_times, beat_times, bidx)
    return dict(times=on_times, beats=on_beats, tempo=tempo)

def summarize(x):
    if len(x)==0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, iqr=np.nan, cv=np.nan)
    q25,q75 = np.percentile(x,[25,75])
    std = np.std(x, ddof=1) if len(x)>1 else 0.0
    mean = float(np.mean(x))
    return dict(n=len(x), mean=mean, median=float(np.median(x)),
                std=float(std), iqr=float(q75-q25),
                cv=float(std/(mean+1e-12)) if mean else np.nan)

def align_ioi_dtw(midi_ioi_b, audio_ioi_b):
    if len(midi_ioi_b)==0 or len(audio_ioi_b)==0:
        return []
    M = midi_ioi_b[:, None]
    A = audio_ioi_b[None, :]
    C = np.abs(np.log((A+1e-9)/(M+1e-9)))
    D, wp = librosa.sequence.dtw(C=C)
    wp = wp[::-1]
    pairs = []
    last_i,last_j = -1,-1
    for (i,j) in wp:
        if i>last_i and j>last_j:
            pairs.append((i,j))
            last_i, last_j = i,j
    pairs = sorted(set(pairs))
    return pairs

def compare(midi_path, audio_path, allowed_vals=(0.5,1.0,2.0,4.0),
            max_rel_err=0.35, ignore_long_beats=None, plots=False, auto_allowed=False):
    m = pm.PrettyMIDI(midi_path)
    if auto_allowed:
        inferred = infer_allowed_from_midi(m)
        if inferred:
            allowed_vals = tuple(inferred)
            print(f"[auto] inferred allowed IOIs (beats): {allowed_vals}")
        else:
            print("[auto] could not infer, using defaults")

    midi_on_beats, inst = extract_midi_onsets_beats(m)
    midi_ioi_b = compute_ioi(midi_on_beats)
    if ignore_long_beats:
        midi_ioi_b = midi_ioi_b[midi_ioi_b <= ignore_long_beats]
    midi_expected_b, midi_assign_ok = quantize_expected(midi_ioi_b, allowed_vals, max_rel_err=max_rel_err)

    A = load_audio_onsets_beats(audio_path)
    audio_ioi_b = compute_ioi(A["beats"])
    if ignore_long_beats:
        audio_ioi_b = audio_ioi_b[audio_ioi_b <= ignore_long_beats]

    pairs = align_ioi_dtw(midi_ioi_b, audio_ioi_b)

    rows = []
    for i,j in pairs:
        exp = midi_expected_b[i]
        ok  = midi_assign_ok[i]
        played = audio_ioi_b[j]
        ratio = played/exp if ok else np.nan
        rows.append(dict(midi_idx=i, audio_idx=j,
                         midi_ioi_beats=midi_ioi_b[i],
                         expected_beats=exp, expected_assigned=bool(ok),
                         audio_ioi_beats=played,
                         ratio_played_over_expected=ratio))
    df = pd.DataFrame(rows)

    ratios_all = df.loc[df["expected_assigned"], "ratio_played_over_expected"].dropna().values
    overall = dict(
        file_midi=midi_path, file_audio=audio_path,
        instrument_program=int(inst.program),
        midi_ioi_beats_stats=summarize(midi_ioi_b),
        audio_ioi_beats_stats=summarize(audio_ioi_b),
        ratio_all_stats=summarize(ratios_all),
        matched_pairs=int(len(df)),
        tempo_est_bpm=float(A["tempo"])
    )

    per_cat = []
    for v in allowed_vals:
        mask = (df["expected_assigned"]) & np.isclose(df["expected_beats"], v)
        r = df.loc[mask, "ratio_played_over_expected"].dropna().values
        per_cat.append(dict(expected_beats=v,
                            count=int(np.sum(mask)),
                            mean=float(np.mean(r)) if len(r) else np.nan,
                            median=float(np.median(r)) if len(r) else np.nan,
                            std=float(np.std(r, ddof=1)) if len(r)>1 else 0.0,
                            cv=float(np.std(r, ddof=1)/(np.mean(r)+1e-12)) if len(r)>1 else np.nan))

    print("\n=== MIDI↔AUDIO IOI CONSISTENCY ===")
    print(pd.Series(overall, dtype=object))
    print("\nPer-category (played / expected, 1.0 ideal):")
    print(pd.DataFrame(per_cat))

    if plots:
        import matplotlib.pyplot as plt
        cats, vals = [], []
        for v in allowed_vals:
            r = df.loc[(df["expected_assigned"]) & np.isclose(df["expected_beats"], v), "ratio_played_over_expected"].dropna().values
            if len(r):
                cats.append(f"{v} beats")
                vals.append(r)
        if vals:
            plt.figure()
            plt.title("Per-Category Played/Expected")
            plt.boxplot(vals, labels=cats, showmeans=True)
            plt.axhline(1.0, linestyle="--")
            plt.ylabel("ratio")
            plt.tight_layout(); plt.show()

    out_csv = "ioi_compare_matches.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved match details → {out_csv}")
    return df, overall

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare MIDI baseline IOIs to AUDIO performance IOIs.")
    ap.add_argument("midi_path")
    ap.add_argument("audio_path")
    ap.add_argument("--allowed", type=str, default="0.5,1.0,2.0,4.0",
                    help="Comma-separated expected IOIs in beats (e.g., '0.5,1.0,2.0').")
    ap.add_argument("--auto-allowed", action="store_true",
                    help="Infer rhythmic IOI categories from the MIDI.")
    ap.add_argument("--max-rel-err", type=float, default=0.35,
                    help="Max relative error to assign an IOI to the nearest allowed value.")
    ap.add_argument("--ignore-long-beats", type=float, default=None,
                    help="Ignore IOIs longer than this many beats (filters rests).")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    allowed_vals = tuple(float(x.strip()) for x in args.allowed.split(",") if x.strip())
    compare(args.midi_path, args.audio_path,
            allowed_vals=allowed_vals,
            max_rel_err=args.max_rel_err,
            ignore_long_beats=args.ignore_long_beats,
            plots=args.plots,
            auto_allowed=args.auto_allowed)
