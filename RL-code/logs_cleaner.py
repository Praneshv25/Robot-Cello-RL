import time
import pandas as pd
import socket
import mido
from mido import MidiFile, tempo2bpm, bpm2tempo
import sys
import numpy as np



CLEF = "bass" # "bass" or "tenor"
MIDI_FILE_PATH = "/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/MIDI-Files/minuet_no_2v2.mid" 
SONG_SCRIPT_TEMPLATE = "/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/URScripts/song.script"
OUTPUT_LOG_FILENAME = "/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Data-Files/Big-Logs/minuet_no_2v2-log-detailed.csv"
BOWING_FILE = "None"
DEFAULT_TEMPO_BPM = 120 # Default tempo in BPM if not specified in MIDI

def get_note_name(note_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note = note_names[note_number % 12]
    return f"{note}{octave}"

def read_bowing_file(bowing_file):
    # (Keep your existing read_bowing_file function)
    bowing_dict = {}
    try:
        with open(bowing_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    try:
                        index = int(parts[0])
                        bowing_dict[index] = parts[1].strip().strip("'")
                    except ValueError:
                        print(f"Warning: Skipping invalid line in bowing file: {line.strip()}")
    except FileNotFoundError:
        print(f"Warning: Bowing file not found at {bowing_file}. Using default bowing.")
        return None
    return bowing_dict

def parse_midi(file_path, bowing_file="None", clef="bass"):
    try:
        midi = MidiFile(file_path)
    except FileNotFoundError:
        print(f"❌ Error: MIDI file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error parsing MIDI file {file_path}: {e}")
        sys.exit(1)

    note_events = []
    bowing_dict = read_bowing_file(bowing_file) if bowing_file != "None" else None

    # Cello string mapping based on MIDI note number
    def get_cello_string(note_number):
        if note_number >= 57: return 'A' # A3
        elif note_number >= 50: return 'D' # D3
        elif note_number >= 43: return 'G' # G2
        else: return 'C' # C2

    last_bow = "down"
    index = 0
    tempo = bpm2tempo(DEFAULT_TEMPO_BPM) # Default tempo in microseconds per beat
    ticks_per_beat = midi.ticks_per_beat if midi.ticks_per_beat > 0 else 480 # Common default

    print(f"MIDI Ticks Per Beat: {ticks_per_beat}")

    absolute_time_sec = 0.0 # Keep track of time in seconds

    for i, track in enumerate(midi.tracks):
        print(f"Processing Track {i}: {track.name}")
        raw_notes = []
        active_notes = {} # Store start time (in seconds) and original note number
        current_tick = 0

        for msg in track:
            # --- Calculate time delta in seconds ---
            delta_ticks = msg.time
            delta_seconds = mido.tick2second(delta_ticks, ticks_per_beat, tempo)
            absolute_time_sec += delta_seconds
            current_tick += delta_ticks

            # --- Update Tempo if message found ---
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
                print(f"  Tempo changed to {tempo2bpm(tempo):.2f} BPM at tick {current_tick} ({absolute_time_sec:.3f}s)")

            # --- Handle Note On ---
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {'start_sec': absolute_time_sec, 'start_tick': current_tick}

            # --- Handle Note Off (or Note On with velocity 0) ---
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_info = active_notes.pop(msg.note)
                    start_time_sec = start_info['start_sec']
                    start_time_tick = start_info['start_tick']
                    end_time_sec = absolute_time_sec
                    end_time_tick = current_tick

                    duration_sec = end_time_sec - start_time_sec
                    duration_ticks = end_time_tick - start_time_tick
                    # Use duration_sec if available, otherwise estimate from ticks
                    if duration_sec < 1e-6: # Avoid zero/negative duration if times are identical
                         duration_sec = mido.tick2second(duration_ticks, ticks_per_beat, tempo)


                    # Apply clef adjustment for string mapping
                    mapping_note = msg.note - 12 if clef == "tenor" else msg.note
                    string = get_cello_string(mapping_note)

                    # Determine bowing
                    if bowing_dict:
                        if index in bowing_dict:
                            bowing = bowing_dict[index]
                            if "-s" in bowing: # Slurred note
                                bowing = last_bow + "-s" # Inherit previous direction for slur
                            else:
                                last_bow = bowing # Update last explicit bow direction
                        else:
                            # Default alternation if index not in file
                            bowing = "up" if last_bow == "down" else "down"
                            last_bow = bowing
                    else:
                        # Default alternation if no file provided
                        bowing = "up" if last_bow == "down" else "down"
                        last_bow = bowing

                    raw_notes.append({
                        'number': msg.note,
                        'note': get_note_name(msg.note),
                        'duration_sec': duration_sec,
                        'string': string,
                        'start_time_sec': start_time_sec,
                        'end_time_sec': end_time_sec,
                        'bowing': bowing,
                        'is_transition': False, # Mark as a real note
                        'event_index': index # Store original index for reference
                    })
                    index += 1
                else:
                     # Note off received for a note not actively tracked (might happen)
                     # print(f"Warning: Received note_off for inactive note {msg.note} at {absolute_time_sec:.3f}s")
                     pass

        # Sort notes by start time (important if MIDI isn't strictly ordered)
        raw_notes.sort(key=lambda x: x['start_time_sec'])

        # Add transitions between notes on different strings
        processed_notes_with_transitions = []
        last_note_end_time = 0.0
        for i, note in enumerate(raw_notes):
             # Check for gap between notes, could represent silence or time for transition
             time_gap = note['start_time_sec'] - last_note_end_time

             # Add the note itself
             processed_notes_with_transitions.append(note)

             # Check if next note exists and requires a string transition
             if i + 1 < len(raw_notes):
                 next_note = raw_notes[i+1]
                 current_string = note['string']
                 next_string = next_note['string']

                 if next_string != current_string:
                     # Estimate transition start/end time - place it right after the current note ends
                     transition_start_time = note['end_time_sec']
                     # Estimate a fixed duration for transition, or use part of the gap if available
                     transition_duration = min(0.2, time_gap) if time_gap > 0.01 else 0.2 # Example: 200ms or use gap
                     transition_end_time = transition_start_time + transition_duration

                     processed_notes_with_transitions.append({
                         'number': 'transition',
                         'note': f"transition {current_string}->{next_string}",
                         'duration_sec': transition_duration,
                         'string': f"{current_string}-{next_string}",
                         'start_time_sec': transition_start_time,
                         'end_time_sec': transition_end_time,
                         'bowing': "transition",
                         'is_transition': True, # Mark as transition
                         'event_index': -1 # No specific note index
                     })
                     last_note_end_time = transition_end_time # Update end time after transition
                 else:
                    last_note_end_time = note['end_time_sec'] # Update end time after same-string note
             else:
                 last_note_end_time = note['end_time_sec'] # Update for the last note

        note_events.extend(processed_notes_with_transitions)

    # Final sort ensures transitions are correctly placed if tracks were processed out of order
    note_events.sort(key=lambda x: x['start_time_sec'])
    print(f"Parsed {len(note_events)} events (notes and transitions).")
    return note_events


# --- URScript Generation ---
script_funcs = {
    "A": "a_bow", "D": "d_bow", "G": "g_bow", "C": "c_bow",
    "A-D": "a_to_d", "D-A": "d_to_a", "D-G": "d_to_g", "G-D": "g_to_d",
    "A-G": "a_to_g", "G-A": "g_to_a", "D-C": "d_to_c", "C-D": "c_to_d",
    "G-C": "g_to_c", "C-G": "c_to_g"
}

def get_function_sequence(note_sequence):
    res = ""
    # Use bowing from MIDI parsing result
    for note in note_sequence:
        function = script_funcs.get(note["string"], None)
        if not function:
            print(f"Warning: No script function found for string '{note['string']}'")
            continue

        # Use duration in seconds from MIDI parsing
        # Note: URScript might interpret time differently; scaling might be needed.
        # The '2' multiplier here seems arbitrary - removing it to use MIDI duration directly.
        # Consider if your URScript functions expect a scaled duration.
        note_duration_sec = note["duration_sec"]

        if note['is_transition']:
            res += f"{function}()\n  "
        else:
            # URScript expects boolean for bowing? True for up, False for down? Adjust as needed.
            bowing_value = "True" if "up" in note["bowing"] else "False"
            # Make sure duration isn't negative or extremely small
            safe_duration = max(0.01, note_duration_sec)
            res += f"{function}({bowing_value}, {safe_duration:.4f})\n  stay()\n  "
    return res
def clean_log(log_filename, note_sequence):
    """
    Cleans and enriches a robot log CSV file by associating log entries
    with corresponding MIDI note events based on time. It preserves
    the existing 'event_label' (e.g., Start/Intermediate/End flags)
    and appends MIDI-derived information.

    Args:
        log_filename (str): The path to the input CSV log file.
        note_sequence (list): A list of dictionaries, where each dictionary
                              represents a parsed MIDI note/transition event
                              with 'start_time_sec', 'end_time_sec', 'note',
                              'string', 'bowing', 'is_transition', etc.
    """
    try:
        df = pd.read_csv(log_filename)

        # Initialize new columns to store MIDI event information
        df['midi_event_type'] = None
        df['midi_note_name'] = None
        df['midi_string_info'] = None
        df['midi_bowing'] = None
        df['midi_event_index'] = None
        df['midi_start_time_sec'] = None
        df['midi_end_time_sec'] = None
        df['midi_duration_sec'] = None
        # num transitions is num note_sequence events where 'bowing' is 'transition
        num_transitions = sum(1 for note in note_sequence if note['is_transition'])
        total_time = 0.2 * num_transitions 
        
        print(f"Cleaning log file: {log_filename}")
        print(f"Number of log rows: {len(df)}")
        print(f"Number of MIDI events: {len(note_sequence)}")

        # Iterate through each row of the DataFrame
        for i, row in df.iterrows():
            current_log_time = row['time_elapsed_sec']

            # Find the MIDI event that spans this log entry's time
            # Assumes note_sequence is sorted by 'start_time_sec'
            found_midi_event = None
            for midi_event in note_sequence:
                # Check if the log time falls within the MIDI event's duration
                if midi_event['start_time_sec'] <= current_log_time < midi_event['end_time_sec']:
                    found_midi_event = midi_event
                    break
            
            if found_midi_event:
                # Populate the new columns with MIDI event details
                df.at[i, 'midi_event_type'] = "Transition" if found_midi_event['is_transition'] else "Note"
                df.at[i, 'midi_note_name'] = found_midi_event['note'] # e.g., 'C4', 'transition A->D'
                df.at[i, 'midi_string_info'] = found_midi_event['string']
                df.at[i, 'midi_bowing'] = found_midi_event['bowing']
                df.at[i, 'midi_event_index'] = found_midi_event['event_index']
                df.at[i, 'midi_start_time_sec'] = found_midi_event['start_time_sec']
                df.at[i, 'midi_end_time_sec'] = found_midi_event['end_time_sec']
                df.at[i, 'midi_duration_sec'] = found_midi_event['duration_sec']

                # Append MIDI-derived information to the *existing* 'event_label'
                # Check if event_label exists and is not NaN/None
                current_label = str(row['event_label']) if pd.notna(row['event_label']) else ""
                
                midi_info_suffix = (
                    f" (MIDI: {found_midi_event['note']} on {found_midi_event['string']} "
                    f"| Bow: {found_midi_event['bowing']})"
                )
                
                # Append only if a valid current_label exists, otherwise just use the MIDI info
                if current_label.strip():
                    df.at[i, 'event_label'] = current_label + midi_info_suffix
                else:
                    df.at[i, 'event_label'] = midi_info_suffix.strip(' ()') # Remove leading space and parentheses if it's the only content
            else:
                # For rows that don't match any MIDI event (e.g., idle periods)
                df.at[i, 'midi_event_type'] = "Idle/Unmatched"
                # The original 'event_label' for these rows will remain unchanged.
                
        # Save the updated DataFrame back to the same CSV file (or a new one)
        df.to_csv(log_filename, index=False, float_format='%.6f') # Use float_format for consistent decimals
        print(f"✅ Cleaned and enriched log file '{log_filename}'.")

    except FileNotFoundError:
        print(f"❌ Error: Log file not found at '{log_filename}'")
    except KeyError as e:
        print(f"❌ Error: Missing expected column in log file: {e}. "
              f"Please ensure 'time_elapsed_sec' and 'event_label' exist.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while cleaning log file '{log_filename}': {e}")
        import traceback
        traceback.print_exc()
'''
def clean_log(log_filename, note_sequence):
    try:
        df = pd.read_csv(log_filename)
        # iterate through all rows of DataFrame and assign corresponding note event 
        curr_event_idx = 0
        start_this_event = True
        end_this_event = False
        print("This is the note sequence:")
        print(note_sequence)
        for row in df.itertuples():
            if curr_event_idx >= len(note_sequence):
                break  # No more note events to label
            event_label = ""
            if 'current_event_type' in df.columns and row.current_event_type == 'Note':
                # Find the corresponding note in the note_sequence
                if start_this_event:
                    df.at[row.Index, 'event_label'] += f"Start {script_funcs.get((note_sequence[curr_event_idx])['string'], None)}"
                    start_this_event = False
                    if ((row.Index + 2) > len(df)) or df.at[row.Index + 2, 'current_event_type'] != 'Note' or df.at[row.Index + 2, 'bow_direction'] != df.at[row.Index, 'bow_direction'] or df.at[row.Index + 2, 'current_string'] != df.at[row.Index, 'current_string']:
                        end_this_event = True
                elif end_this_event:
                    df.at[row.Index, 'event_label'] += f"End {script_funcs.get((note_sequence[curr_event_idx])['string'], None)}"
                    end_this_event = False
                    start_this_event = True
                    curr_event_idx += 1
                else:
                    df.at[row.Index, 'event_label'] += f"Intermediate {script_funcs.get((note_sequence[curr_event_idx])['string'], None)}"
                    if ((row.Index + 2) > len(df)) or df.at[row.Index + 2, 'current_event_type'] != 'Note' or df.at[row.Index + 2, 'bow_direction'] != df.at[row.Index, 'bow_direction'] or df.at[row.Index + 2, 'current_string'] != df.at[row.Index, 'current_string']:
                        end_this_event = True
            elif 'current_event_type' in df.columns and row.current_event_type == 'Transition':
                # Handle transitions if needed
                if start_this_event:
                    df.at[row.Index, 'event_label'] += f"Start Transition"
                    start_this_event = False
                    if ((row.Index + 2) > len(df)) or df.at[row.Index + 2, 'current_event_type'] != 'Transition' or df.at[row.Index + 2, 'bow_direction'] != df.at[row.Index, 'bow_direction'] or df.at[row.Index + 2, 'current_string'] != df.at[row.Index, 'current_string']:
                        end_this_event = True
                elif end_this_event:
                    df.at[row.Index, 'event_label'] += f"End Transition"
                    end_this_event = False
                    start_this_event = True
                    curr_event_idx += 1
                else:
                    df.at[row.Index, 'event_label'] += f"Intermediate Transition"
                    if ((row.Index + 2) > len(df)) or df.at[row.Index + 2, 'current_event_type'] != 'Transition' or df.at[row.Index + 2, 'bow_direction'] != df.at[row.Index, 'bow_direction'] or df.at[row.Index + 2, 'current_string'] != df.at[row.Index, 'current_string']:
                        end_this_event = True
            else:
                print(f"Warning: Row {row.Index} has unexpected current_event_type '{row.current_event_type}' or missing 'event_label' column.")
        df.to_csv(log_filename, index=False)
        print(f"✅ Cleaned log file '{log_filename}' timestamps.")
    except Exception as e:
        print(f"❌ Error cleaning log file '{log_filename}': {e}")

'''

# --- Script Execution ---
if __name__ == "__main__":
    print("Parsing MIDI file...")
    note_sequence_timed = parse_midi(MIDI_FILE_PATH, BOWING_FILE, CLEF)

    if not note_sequence_timed:
        print("❌ No note events parsed from MIDI. Exiting.")
        sys.exit(1)

    # print("\n--- Parsed Note Sequence (with seconds) ---")
    # for note in note_sequence_timed[:10]: # Print first few notes
    #     print(f"  {note['start_time_sec']:.3f}s - {note['end_time_sec']:.3f}s ({note['duration_sec']:.3f}s): "
    #           f"{note['note']} ({note['string']}) Bow: {note['bowing']} Transition: {note['is_transition']}")
    # print("...\n")


    print("Generating URScript function sequence...")
    function_sequence = get_function_sequence(note_sequence_timed)

    print("Loading URScript template...")
    try:
        with open(SONG_SCRIPT_TEMPLATE, "r") as f:
            script_template = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Song script template not found at {SONG_SCRIPT_TEMPLATE}")
        sys.exit(1)

    # Inject the generated function calls into the template
    final_script = script_template.replace("# $$$ CODE HERE $$$", function_sequence)

    # print("\n--- Generated URScript (snippet) ---")
    # print(function_sequence[:500] + "...") # Print beginning of generated part
    # print("---")
    print(function_sequence)
    print(note_sequence_timed)
    # Save the generated script for inspection (optional)
    try:
        with open('generated_cello_script.txt', "w") as test_file:
            test_file.write(final_script)
        print("✅ Saved full generated URScript to 'generated_cello_script.txt'")
    except Exception as e:
        print(f"⚠️ Warning: Could not save generated script file: {e}")

    clean_log(OUTPUT_LOG_FILENAME, note_sequence_timed)