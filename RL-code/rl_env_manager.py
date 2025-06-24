import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


def load_rl_env_from_dot_env():
    """
    Searches the parent directory for .env file and call load_env_file with it.

    Expected to used in another script as following:
    ```python
        from rl_env_manager import load_rl_env_from_dot_env

        # Load environment variables
        env_vars = load_rl_env_from_dot_env()

        trajectory = extract_joint_angles(
            env_vars.get("TRAJ_LOG_CSV_PATH")
        )
    ```

    :return: dictionary of parsed environment variable
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dot_env_path = os.path.abspath(os.path.join(script_dir, '..', '.env'))
    return load_env_file(dot_env_path)


def load_env_file(filepath):
    """
    Internal helper for parsing the `.env` file in the project root directory
    and creating a dictionary to access them.

    :return: dictionary of parsed environment variable
    :raises ValueError: if any of the environment variable is empty
    """

    env_dict = {}
    if not os.path.isfile(filepath):
        return env_dict

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env_dict[key] = value
    return env_dict

        env_vars = {
            "TRAJ_LOG_CSV_PATH": os.getenv("TRAJ_LOG_CSV_PATH"),
            "NOTE_SEQ_MIDI_PATH": os.getenv("NOTE_SEQ_MIDI_PATH"),
            "SCENE_XML_PATH": os.getenv("SCENE_XML_PATH"),
            "MODEL_ZIP_PATH": os.getenv("MODEL_ZIP_PATH")
        }

        if not all(env_vars.items()):
            raise ValueError("Missing one of the environment variables")

        return env_vars

    except Exception as e:
        print(f"Error loading environment variables: {e}")
        return None
