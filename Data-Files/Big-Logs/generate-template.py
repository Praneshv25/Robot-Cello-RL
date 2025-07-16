import pandas as pd

def process_log(input_csv_path, output_csv_path):
    """
    Processes the log CSV file to add a 'current_note_event' column
    and modify the 'event_label' column with 'Starting', 'Playing', 'Ending' messages.

    Args:
        input_csv_path (str): Path to the original input CSV file.
        output_csv_path (str): Path where the modified CSV file will be saved.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return

    df['current_note_event'] = df['current_string']

    # Create an empty list to store the new log messages for event_label
    modified_event_labels = []

    # Iterate through the DataFrame to apply the logic for event_label messages
    for i, row in df.iterrows():
        event_label = row['event_label']
        current_note_event = row['current_note_event']

        # Extract the actual event name (e.g., 'd_bow' from 'START d_bow')
        parts = event_label.split(' ', 1)
        event_type = parts[0] # 'START' or 'END'
        event_name = parts[1] if len(parts) > 1 else ""

        message = ""

        if event_type == 'START':
            # Check if it's the start of a new sequence of current_note_event
            # or if it's the very first row
            if i == 0 or df.loc[i-1, 'current_note_event'] != current_note_event:
                message = f"Starting {event_name}..."
            else:
                message = f"Playing {event_name}..."
        elif event_type == 'END':
            # Check if it's the end of the current sequence of current_note_event
            # or if it's the very last row
            if i == len(df) - 1 or df.loc[i+1, 'current_note_event'] != current_note_event:
                message = f"Ending {event_name}..."
            else:
                message = f"Playing {event_name}..."
        else:
            # For any other event_type not explicitly handled (e.g., if there are other types)
            message = event_label # Keep original if not 'START' or 'END'

        modified_event_labels.append(message)

    # Replace the original 'event_label' column with the new modified messages
    df['event_label'] = modified_event_labels

    # Save the modified DataFrame to the specified output CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Processed file saved to {output_csv_path}")


input_file = '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Data-Files/Big-Logs/allegro-log-detailed.csv'
output_file = 'allegro-log-detailed-modified.csv'
process_log(input_file, output_file)