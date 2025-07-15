#!/usr/bin/env python3
"""
fix_log_event_labels.py
=======================

Utility to correct the *event_label* column in a detailed robot execution log
created by **robot_runner_detailed_logs.py** so that each row is tagged with the
actual musical note that was active at that timestamp.

It also adds two convenience columns:

* ``note_index`` – index of the note within the original ``note_sequence``  
* ``note_name``  – human‑readable note (e.g. ``G4``)  

How it works
------------
1. Dynamically imports ``parse_midi`` from *robot_runner_detailed_logs.py* (so we
   don’t duplicate parsing logic).
2. Parses the original MIDI file (and optional bowing‑text file) to recover the
   exact start time and duration of every note in ``note_sequence``.
3. Walks through the CSV, assigning each row to the note whose time window it
   falls inside (with a small numerical tolerance for safety).
4. Overwrites the existing ``event_label`` with the note name and writes the
   enhanced CSV back to disk.

Basic usage
-----------

.. code:: bash

    python fix_log_event_labels.py \
        --csv    minuet_no_2v2-log-detailed.csv \
        --output minuet_no_2v2-log-detailed-fixed.csv \
        --script robot_runner_detailed_logs.py \
        --midi   MIDI-Files/minuet_no_2v2.mid

Optional flags: ``--bowing-file`` and ``--clef`` pass straight through to
``parse_midi`` if you need them.

"""

import pandas as pd
import argparse
import importlib.util
import sys
from pathlib import Path

TOLERANCE = 1e-6  # seconds


def _load_parse_midi(script_path: Path):
    """Dynamically import *parse_midi* from the given script file."""
    spec = importlib.util.spec_from_file_location("robot_runner_detailed_logs", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, "parse_midi"):
        raise AttributeError(f"{script_path} does not define a 'parse_midi' function.")
    return module.parse_midi


def _load_note_sequence(parse_midi_fn, midi_path: Path, bowing_file: str, clef: str):
    note_seq = parse_midi_fn(str(midi_path), bowing_file, clef)
    if not note_seq:
        raise RuntimeError("parse_midi returned an empty note_sequence – nothing to label!")
    # Ensure chronological order
    return sorted(note_seq, key=lambda n: n["start_time_sec"])


def _apply_labels(df: pd.DataFrame, note_seq):
    # Prepare new columns
    df["note_index"] = -1
    df["note_name"] = pd.NA

    # Vectorised labelling per note
    for idx, note in enumerate(note_seq):
        start = note["start_time_sec"] - TOLERANCE
        end = note["start_time_sec"] + note["duration_sec"] + TOLERANCE
        mask = (df["time_elapsed_sec"] >= start) & (df["time_elapsed_sec"] < end)
        df.loc[mask, "note_index"] = idx
        df.loc[mask, "note_name"] = note["note"]

    # Overwrite event_label where note_name is known
    df.loc[df["note_name"].notna(), "event_label"] = df.loc[df["note_name"].notna(), "note_name"]
    return df


def main():
    parser = argparse.ArgumentParser(description="Fix event_label and add note mapping to robot log CSV.")
    parser.add_argument("--csv", required=True, help="Path to input CSV produced by robot_runner_detailed_logs.py")
    parser.add_argument("--output", required=True, help="Path to write the fixed CSV")
    parser.add_argument("--script", required=True, help="Path to robot_runner_detailed_logs.py (to import parse_midi)")
    parser.add_argument("--midi", required=True, help="Path to the original MIDI file used for this run")
    parser.add_argument("--bowing-file", default="None", help="Optional bowing file (pass 'None' to skip)")
    parser.add_argument("--clef", default="bass", help="Clef to hand to parse_midi (default='bass')")

    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    script_path = Path(args.script).expanduser().resolve()
    midi_path = Path(args.midi).expanduser().resolve()

    # --- Load resources -----------------------------------------------------
    parse_midi_fn = _load_parse_midi(script_path)
    note_seq = _load_note_sequence(parse_midi_fn, midi_path, args.bowing_file, args.clef)
    df = pd.read_csv(csv_path)

    # --- Relabel ------------------------------------------------------------
    df_fixed = _apply_labels(df, note_seq)

    # --- Save ---------------------------------------------------------------
    df_fixed.to_csv(output_path, index=False)
    print(f"✅ Wrote fixed CSV with {len(df_fixed):,} rows to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ {type(exc).__name__}: {exc}")
        sys.exit(1)

# /Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/.venv/bin/python /Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/logs_cleaner_2.py --csv /Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Data-Files/Big-Logs/minuet_no_2v2-log-detailed.csv --output minuet_no_2v2-log-detailed-fixed.csv --script RL-code/robot_runner_detailed_logs.py --midi /Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/MIDI-Files/minuet_no_2v2.mid