import time
import pandas as pd
import socket
import mido
from mido import MidiFile, tempo2bpm, bpm2tempo
import sys
import numpy as np
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import logging # Added for example_control_loop's logging setup

# ================================
# Configuration (Keep existing, add DASHBOARD_PORT)
# ================================
ROBOT_IP = "10.165.11.242"
ROBOT_PORT = 30004  # RTDE Port
UR_PRIMARY_PORT = 30002 # Primary Interface Port for URScript
DASHBOARD_PORT = 29999 # Dashboard Server Port

CLEF = "bass"
# Update these paths to your actual file locations
CONFIG_FILENAME = "/Users/samanthasudhoff/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/RTDE_Python_Client_Library/examples/cello_config.xml"
MIDI_FILE_PATH = "/Users/samanthasudhoff/Documents/GitHub/Robot-Cello-ResidualRL/MIDI-Files/allegro.mid"
BOWING_FILE = "None"
SONG_SCRIPT_TEMPLATE = "/Users/samanthasudhoff/Documents/GitHub/Robot-Cello-ResidualRL/URScripts/song.script"
OUTPUT_LOG_FILENAME = "allegro-detailed-test-irl.csv"
DEFAULT_TEMPO_BPM = 120

# ================================
# Dashboard Helpers (Copied from example_control_loop.py)
# ================================
def dashboard_command(cmd):
    try:
        with socket.create_connection((ROBOT_IP, DASHBOARD_PORT), timeout=2) as dash:
            dash.sendall((cmd + "\n").encode())
            resp = dash.recv(1024).decode().strip()
            print(f"📟 Dashboard: {resp}")
            return resp
    except Exception as e:
        print(f"⚠️ Dashboard command failed: {e}")
        return None

def dashboard_play_with_wait():
    resp = dashboard_command("play")
    if resp and "Failed" in resp:
        print("⏳ Waiting for program to be ready...")
        for _ in range(10):
            state = dashboard_command("programState")
            if state and "READY" in state:
                dashboard_command("play")
                return
            time.sleep(0.5)
        print("▶️ Program started.")
    else:
        print("▶️ Program started (or already running).")


# ================================
# Setup Logging (Copied from example_control_loop.py)
# ================================
logging.getLogger().setLevel(logging.INFO)

# ================================
# RTDE Connection Setup (Modified to align with example_control_loop.py)
# ================================
con = None # Initialize con to None
try:
    conf = rtde_config.ConfigFile(CONFIG_FILENAME)
    # Ensure recipe names match your XML. Assuming "state", "setp", "watchdog" from example_control_loop.py
    state_names, state_types = conf.get_recipe("state")
    setp_names, setp_types = conf.get_recipe("setp") # Added
    watchdog_names, watchdog_types = conf.get_recipe("watchdog") # Added

    con = rtde.RTDE(ROBOT_IP, ROBOT_PORT)

except FileNotFoundError:
    print(f"❌ Error: RTDE Configuration file not found at {CONFIG_FILENAME}")
    sys.exit(1)
except KeyError as e:
    print(f"❌ Error: Required recipe ('state', 'setp', or 'watchdog') not found or missing fields in {CONFIG_FILENAME}. Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error setting up RTDE configuration: {e}")
    sys.exit(1)

# --- Data Log ---
data_log = []
rtde_running = False
cntrl_c = False

# --- Helper Functions (No changes needed here for RTDE logic) ---
def interpret_flag(flag):
    mapping = {
        1: "START a_bow", 2: "END a_bow", 3: "START d_bow", 4: "END d_bow",
        5: "START g_bow", 6: "END g_bow", 7: "START c_bow", 8: "END c_bow",
        101: "START a_to_d", 102: "END a_to_d", 103: "START d_to_a", 104: "END d_to_a",
        105: "START d_to_g", 106: "END d_to_g", 107: "START g_to_d", 108: "END g_to_d",
        109: "START a_to_g", 110: "END a_to_g", 111: "START g_to_a", 112: "END g_to_a",
        113: "START d_to_c", 114: "END d_to_c", 115: "START c_to_d", 116: "END c_to_d",
        117: "START g_to_c", 118: "END g_to_c", 119: "START c_to_g", 120: "END c_to_g",
        0: "IDLE/BETWEEN"
    }
    if flag == -1: return "INITIALIZING"
    return mapping.get(flag, f"Unknown ({flag})")

def get_note_name(note_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note = note_names[note_number % 12]
    return f"{note}{octave}"

def read_bowing_file(bowing_file):
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

    def get_cello_string(note_number):
        if note_number >= 57: return 'A'
        elif note_number >= 50: return 'D'
        elif note_number >= 43: return 'G'
        else: return 'C'

    last_bow = "down"
    index = 0
    tempo = bpm2tempo(DEFAULT_TEMPO_BPM)
    ticks_per_beat = midi.ticks_per_beat if midi.ticks_per_beat > 0 else 480

    print(f"MIDI Ticks Per Beat: {ticks_per_beat}")

    absolute_time_sec = 0.0

    for i, track in enumerate(midi.tracks):
        print(f"Processing Track {i}: {track.name}")
        raw_notes = []
        active_notes = {}
        current_tick = 0

        for msg in track:
            delta_ticks = msg.time
            delta_seconds = mido.tick2second(delta_ticks, ticks_per_beat, tempo)
            absolute_time_sec += delta_seconds
            current_tick += delta_ticks

            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
                print(f"  Tempo changed to {tempo2bpm(tempo):.2f} BPM at tick {current_tick} ({absolute_time_sec:.3f}s)")

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {'start_sec': absolute_time_sec, 'start_tick': current_tick}

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_info = active_notes.pop(msg.note)
                    start_time_sec = start_info['start_sec']
                    start_time_tick = start_info['start_tick']
                    end_time_sec = absolute_time_sec
                    end_time_tick = current_tick

                    duration_sec = end_time_sec - start_time_sec
                    duration_ticks = end_time_tick - start_time_tick
                    if duration_sec < 1e-6:
                         duration_sec = mido.tick2second(duration_ticks, ticks_per_beat, tempo)

                    mapping_note = msg.note - 12 if clef == "tenor" else msg.note
                    string = get_cello_string(mapping_note)

                    if bowing_dict:
                        if index in bowing_dict:
                            bowing = bowing_dict[index]
                            if "-s" in bowing:
                                bowing = last_bow + "-s"
                            else:
                                last_bow = bowing
                        else:
                            bowing = "up" if last_bow == "down" else "down"
                            last_bow = bowing
                    else:
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
                        'is_transition': False,
                        'event_index': index
                    })
                    index += 1
                else:
                     pass

        raw_notes.sort(key=lambda x: x['start_time_sec'])

        processed_notes_with_transitions = []
        last_note_end_time = 0.0
        for i, note in enumerate(raw_notes):
             time_gap = note['start_time_sec'] - last_note_end_time

             processed_notes_with_transitions.append(note)

             if i + 1 < len(raw_notes):
                 next_note = raw_notes[i+1]
                 current_string = note['string']
                 next_string = next_note['string']

                 if next_string != current_string:
                     transition_start_time = note['end_time_sec']
                     transition_duration = min(0.2, time_gap) if time_gap > 0.01 else 0.2
                     transition_end_time = transition_start_time + transition_duration

                     processed_notes_with_transitions.append({
                         'number': 'transition',
                         'note': f"transition {current_string}->{next_string}",
                         'duration_sec': transition_duration,
                         'string': f"{current_string}-{next_string}",
                         'start_time_sec': transition_start_time,
                         'end_time_sec': transition_end_time,
                         'bowing': "transition",
                         'is_transition': True,
                         'event_index': -1
                     })
                     last_note_end_time = transition_end_time
                 else:
                    last_note_end_time = note['end_time_sec']
             else:
                 last_note_end_time = note['end_time_sec']

        note_events.extend(processed_notes_with_transitions)

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

        note_duration_sec = note["duration_sec"]

        if note['is_transition']:
            res += f"{function}()\n  "
        else:
            # Change to use 1 for 'up' (True) and 0 for 'down' (False)
            bowing_value_int = 1 if "up" in note["bowing"] else 0 # Changed this line
            safe_duration = max(0.01, note_duration_sec)
            res += f"{function}({bowing_value_int}, {safe_duration:.4f})\n  stay()\n  " # Changed this line
    return res


# --- Main Execution Logic (Heavily Modified) ---

def send_urscript(urscript, speed_scaling, note_sequence_timed):
    global rtde_running, cntrl_c, data_log, con # Make con global to access it

    if not urscript:
        print("❌ Error: No URScript generated!")
        return

    speed_command = f"set_speed({speed_scaling})\n"
    full_urscript = speed_command + urscript

    if len(full_urscript.encode('utf-8')) > 15000:
        print("⚠️ Warning: URScript is large and might exceed robot buffer limits.")

    print("--- URScript to be sent ---")
    # print(full_urscript) # Uncomment to debug
    print("--- End URScript ---")

    print(f"Connecting to RTDE at {ROBOT_IP}:{ROBOT_PORT}...")
    try:
        con.connect() # Use the global 'con' object
        if not con.is_connected: # Check connection status
            print(f"❌ Could not connect to RTDE on {ROBOT_IP}:{ROBOT_PORT}.")
            return
        print(f"✅ RTDE Connected.")

        try:
            con.get_controller_version()
        except Exception as e:
            print(f"⚠️ Warning: Could not get controller version after connect: {e}")

        # --- Setup RTDE Output and Input Recipes (Aligned with example_control_loop.py) ---
        print("Setting up RTDE output recipe...")
        if not con.send_output_setup(state_names, state_types):
             print("❌ Failed to setup RTDE output recipe.")
             con.disconnect()
             return
        print("✅ RTDE output recipe sent.")

        print("Setting up RTDE input recipes (setp, watchdog)...")
        setp = con.send_input_setup(setp_names, setp_types) # Initialize setp
        watchdog = con.send_input_setup(watchdog_names, watchdog_types) # Initialize watchdog
        if setp is None or watchdog is None:
            print("❌ Failed to setup RTDE input recipes.")
            con.disconnect()
            return
        print("✅ RTDE input recipes sent.")

        # --- Initialize input registers (Similar to example_control_loop.py) ---
        # NOTE: Your robot_runner script doesn't use setp for real-time control
        # like example_control_loop does for Cartesian poses.
        # However, you *must* initialize setp and watchdog objects if you defined them
        # in your XML and are sending them. For now, we'll initialize them
        # as dummy values if they are not actively used for control in this script.
        # If your 'setp' recipe in cello_config.xml is for actual_TCP_pose, you might
        # need to send a valid pose here. If it's just a placeholder, 0s are fine.
        # Example: setp.input_double_register_0 = 0.0 # and so on for all 6
        # For simplicity, if not directly controlling pose via RTDE inputs,
        # ensure your robot program doesn't expect these to be active.

        # The watchdog is crucial for keeping the RTDE connection alive and indicating activity.
        watchdog.input_int_register_0 = 0 # Initialize watchdog to 0 (idle)


        # --- Send URScript via Primary Interface ---
        print(f"Connecting to UR Primary Interface at {ROBOT_IP}:{UR_PRIMARY_PORT} to send script...")
        with socket.create_connection((ROBOT_IP, UR_PRIMARY_PORT), timeout=10) as sock:
            print("✅ Connected to Primary Interface.")
            sock.sendall(full_urscript.encode('utf-8'))
            print("✅ Sent URScript.")
            # Give robot a moment to start the program, or use dashboard_play_with_wait
            time.sleep(0.5)

        # Optional: Start the robot program via dashboard (if it's not started by sending the script)
        # dashboard_play_with_wait() # This might interfere if the script auto-runs

        # --- Start RTDE Data Synchronization ---
        if not con.send_start():
            print("❌ Failed to start RTDE data synchronization.")
            con.disconnect()
            return
        print("✅ RTDE data synchronization started.")
        rtde_running = True

        # --- RTDE Data Logging Loop (Modified to include watchdog send) ---
        print("--- Starting RTDE Data Logging (Press Ctrl+C to stop early) ---")
        start_log_time = time.time()

        init_time = -1.0
        last_flag = -1
        last_event_label = "INITIALIZING"
        current_note_info = None
        has_note_changed = False
        current_note_idx = -1

        while rtde_running:
            try:
                state = con.receive()

                if state is None:
                    elapsed_time = time.time() - start_log_time
                    total_estimated_duration = sum(n['duration_sec'] for n in note_sequence_timed) + 5
                    if elapsed_time > total_estimated_duration and len(data_log) > 0: # Only stop if some data was logged
                       print("Estimated script duration exceeded and no more RTDE data. Stopping.")
                       rtde_running = False
                    continue

                if init_time < 0:
                    init_time = state.timestamp
                    print(f"Initial timestamp received: {init_time}")

                current_rtde_time_sec = state.timestamp - init_time

                # --- Flag Change Detection (Event Logging) ---
                current_flag = state.output_int_register_0
                event_label = None
                if current_flag != last_flag:
                    event_label = interpret_flag(current_flag)
                    print(f"[{current_rtde_time_sec:.3f}s] Event: {event_label} (Flag: {current_flag})")
                    last_flag = current_flag
                    last_event_label = event_label

                    if event_label and "START" in event_label:
                       found_note_for_event = False
                       for idx, note_info in enumerate(note_sequence_timed):
                           if note_info['start_time_sec'] >= current_rtde_time_sec - 0.1:
                                if note_info != current_note_info:
                                    has_note_changed = True
                                else:
                                    has_note_changed = False
                                current_note_info = note_info
                                current_note_idx = idx
                                found_note_for_event = True
                                break

                # --- Find Current Note/Transition based on TIME ---
                temp_current_note = None
                temp_current_idx = -1
                search_start_idx = max(0, current_note_idx)

                for idx in range(search_start_idx, len(note_sequence_timed)):
                     note_info = note_sequence_timed[idx]
                     if note_info['start_time_sec'] <= current_rtde_time_sec < note_info['end_time_sec']:
                         temp_current_note = note_info
                         temp_current_idx = idx
                         break

                if temp_current_note:
                    current_note_info = temp_current_note
                    current_note_idx = temp_current_idx

                # --- Calculate Remaining Duration ---
                remaining_duration_sec = 0.0
                current_event_type = "None"
                if current_note_info:
                    remaining_duration_sec = max(0.0, current_note_info['end_time_sec'] - current_rtde_time_sec)
                    current_event_type = "Transition" if current_note_info['is_transition'] else "Note"

                # --- Log Data Point ---
                data_log.append({
                    "timestamp_robot": state.timestamp,
                    "time_elapsed_sec": current_rtde_time_sec,
                    "event_flag": current_flag,
                    "event_label": last_event_label,
                    "current_event_type": current_event_type,
                    "current_note_number": current_note_info.get('number', None) if current_note_info else None,
                    "current_note_name": current_note_info.get('note', None) if current_note_info else None,
                    "current_string": current_note_info.get('string', None) if current_note_info else None,
                    "current_bowing": current_note_info.get('bowing', None) if current_note_info else None,
                    "remaining_duration_sec": remaining_duration_sec,
                    "has_note_changed": has_note_changed,
                    "TCP_pose_x": state.actual_TCP_pose[0],
                    "TCP_pose_y": state.actual_TCP_pose[1],
                    "TCP_pose_z": state.actual_TCP_pose[2],
                    "TCP_pose_rx": state.actual_TCP_pose[3],
                    "TCP_pose_ry": state.actual_TCP_pose[4],
                    "TCP_pose_rz": state.actual_TCP_pose[5],
                    "q_base": state.actual_q[0],
                    "q_shoulder": state.actual_q[1],
                    "q_elbow": state.actual_q[2],
                    "q_wrist1": state.actual_q[3],
                    "q_wrist2": state.actual_q[4],
                    "q_wrist3": state.actual_q[5],
                    "TCP_force_x": state.actual_TCP_force[0],
                    "TCP_force_y": state.actual_TCP_force[1],
                    "TCP_force_z": state.actual_TCP_force[2],
                    "TCP_force_rx": state.actual_TCP_force[3],
                    "TCP_force_ry": state.actual_TCP_force[4],
                    "TCP_force_rz": state.actual_TCP_force[5],
                })

                # --- Send Watchdog Signal (Crucial for active RTDE connection) ---
                # Set watchdog register to 1 if the program is active, 0 otherwise.
                # In your script, the URScript will set output_int_register_0 based on its state.
                # You can echo that back or simply keep the watchdog alive by setting it to 1.
                # For this application, you likely want to keep the watchdog active as long as you're receiving data.
                watchdog.input_int_register_0 = 1 # Keep watchdog active
                con.send(watchdog)


            except socket.timeout:
                 print("RTDE receive socket timeout.")
                 # Consider if timeout means the program has ended on the robot side
                 # If total_estimated_duration is over, then stop.
                 elapsed_time = time.time() - start_log_time
                 total_estimated_duration = sum(n['duration_sec'] for n in note_sequence_timed) + 5
                 if elapsed_time > total_estimated_duration and len(data_log) > 0:
                     print("Estimated script duration exceeded and no more RTDE data (timeout). Stopping.")
                     rtde_running = False
                 continue
            except ConnectionAbortedError:
                 print("RTDE Connection Aborted.")
                 rtde_running = False
                 break
            except Exception as e:
                 print(f"❌ Error in RTDE loop: {e}")
                 import traceback
                 traceback.print_exc()
                 rtde_running = False
                 break

    except socket.timeout:
        print("❌ Socket timeout during connection or sending URScript.")
    except ConnectionRefusedError:
         print(f"❌ Connection refused. Is the robot on ({ROBOT_IP}) and URCap running?")
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt detected. Stopping script and saving data...")
        cntrl_c = True
        rtde_running = False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("--- Execution/Logging Finished ---")
        if con and con.is_connected():
            print("Pausing RTDE stream...")
            try:
                con.send_pause()
                # Set watchdog to 0 on pause/disconnect
                if 'watchdog' in locals() and watchdog is not None:
                     watchdog.input_int_register_0 = 0
                     con.send(watchdog)
            except Exception as e:
                print(f"⚠️ Warning: Exception during send_pause: {e}")
            print("Disconnecting RTDE...")
            try:
                con.disconnect()
            except Exception as e:
                print(f"⚠️ Warning: Exception during disconnect: {e}")
        else:
             print("RTDE connection was not active or already closed.")
        save_data(data_log, OUTPUT_LOG_FILENAME)


def save_data(log_data, filename):
    if not log_data:
        print("No data collected to save.")
        return

    print(f"Saving {len(log_data)} data points to {filename}...")
    try:
        columns = [
            "timestamp_robot", "time_elapsed_sec", "event_flag", "event_label",
            "current_event_type", "current_note_number", "current_note_name",
            "current_string", "remaining_duration_sec",
            "TCP_pose_x", "TCP_pose_y", "TCP_pose_z", "TCP_pose_rx", "TCP_pose_ry", "TCP_pose_rz",
            "q_base", "q_shoulder", "q_elbow", "q_wrist1", "q_wrist2", "q_wrist3",
            "TCP_force_x", "TCP_force_y", "TCP_force_z", "TCP_force_rx", "TCP_force_ry", "TCP_force_rz"
        ]
        filtered_log_data = [{k: d.get(k, None) for k in columns} for d in log_data]

        df_log = pd.DataFrame(filtered_log_data, columns=columns)
        df_log.to_csv(filename, index=False, float_format='%.6f')
        print(f"✅ Successfully saved log data to '{filename}'.")
    except Exception as e:
        print(f"❌ Error saving data log: {e}")


# --- Script Execution ---
if __name__ == "__main__":
    print("Parsing MIDI file...")
    note_sequence_timed = parse_midi(MIDI_FILE_PATH, BOWING_FILE, CLEF)

    if not note_sequence_timed:
        print("❌ No note events parsed from MIDI. Exiting.")
        sys.exit(1)

    print("Generating URScript function sequence...")
    function_sequence = get_function_sequence(note_sequence_timed)

    print("Loading URScript template...")
    try:
        with open(SONG_SCRIPT_TEMPLATE, "r") as f:
            script_template = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Song script template not found at {SONG_SCRIPT_TEMPLATE}")
        sys.exit(1)

    final_script = script_template.replace("# $$$ CODE HERE $$$", function_sequence)

    try:
        with open('generated_cello_script.txt', "w") as test_file:
            test_file.write(final_script)
        print("✅ Saved full generated URScript to 'generated_cello_script.txt'")
    except Exception as e:
        print(f"⚠️ Warning: Could not save generated script file: {e}")

    print("Starting robot execution and data logging...")
    send_urscript(final_script, 0.8, note_sequence_timed)

    print("--- Script Finished ---")