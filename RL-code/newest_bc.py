import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os
import pickle

# URSim output training files (replace with your paths)
# Note: I will gather more data + combine for improved BC results (WIP, esp gathering accurate G/C string data)
CSV_FILES = [
    '/Users/samanthasudhoff/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/allegro-log-detailed-test.csv',
    '/Users/samanthasudhoff/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/twinkle-log-detailed-test.csv',
    '/Users/samanthasudhoff/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/perpetual-log-detailed-test.csv',
    '/Users/samanthasudhoff/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/long-log-detailed-test.csv',
    '/Users/samanthasudhoff/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/minuet-log-detailed-test.csv',
]



# Observation input features from csv data
INPUT_FEATURE_COLS = [
    # curr robot joint states + tcp pose for closed-loop control
    'q_base', 'q_shoulder', 'q_elbow', 'q_wrist1', 'q_wrist2', 'q_wrist3',
    'TCP_pose_x', 'TCP_pose_y', 'TCP_pose_z', 'TCP_pose_rx', 'TCP_pose_ry', 'TCP_pose_rz',
    # musical context
    'time_elapsed_sec',
    'remaining_duration_sec',
    'current_note_number', # remember there is also 'transition'
    'current_string',
    'event_label',
    'event_flag'
]

# will be predicted for curr timestep as BC action
# fix to predict TCP position, not q_pos
TARGET_COLS = [
    'TCP_pose_x', 'TCP_pose_y', 'TCP_pose_z', 'TCP_pose_rx', 'TCP_pose_ry', 'TCP_pose_rz' 
]

# --- Neural Network Parameters (might wanna change) ---
HIDDEN_SIZE = 256
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TEST_SIZE = 0.2 # 20% of data for validation

# --- Dataset and Preprocessing ---
class CelloTrajectoryDataset(Dataset):
    def __init__(self, data_df, feature_cols, target_cols, scalers, one_hot_encoders):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.scalers = scalers
        self.one_hot_encoders = one_hot_encoders

        # Apply scalers and encoders
        X_processed = self._preprocess_features(data_df.copy()) 
        y_processed = self._preprocess_targets(data_df.copy())

        self.X = torch.tensor(X_processed, dtype=torch.float32)
        self.y = torch.tensor(y_processed, dtype=torch.float32)

    def _preprocess_features(self, df_features):
        processed_features = []
        
        # 0 will be for transition events
        df_features['current_note_number'] = pd.to_numeric(df_features['current_note_number'], errors='coerce')
        df_features['current_note_number'] = df_features['current_note_number'].fillna(0) # Fill NaN with 0

        # Numerical features (Robot State & direct Musical Context numerics)
        numerical_cols = [col for col in self.feature_cols if col not in ['current_string', 'event_label', 'event_flag']]
        for col in numerical_cols:
            if col in self.scalers:
                processed_features.append(self.scalers[col].transform(df_features[[col]]))
            else:
                # Should not happen if scalers are fitted correctly for all numerical cols
                processed_features.append(df_features[[col]].values)

        # Categorical features (one-hot encode)
        # current_string
        current_string_encoded = self.one_hot_encoders['current_string'].transform(df_features[['current_string']])
        processed_features.append(current_string_encoded) # Removed .toarray()

        # bow_direction (inferred from event_label)
        bow_direction_df = self._infer_bow_direction(df_features['event_label'])
        bow_direction_encoded = self.one_hot_encoders['bow_direction'].transform(bow_direction_df[['bow_direction']])
        processed_features.append(bow_direction_encoded) # Removed .toarray()

        # is_transition (inferred from event_flag / event_label)
        is_transition_df = self._infer_is_transition(df_features['event_flag'], df_features['event_label'])
        processed_features.append(is_transition_df[['is_transition']].values) # is_transition is already 0/1

        return np.hstack(processed_features)

    def _preprocess_targets(self, df_targets):
        processed_targets = []
        for col in self.target_cols:
            if col in self.scalers:
                processed_targets.append(self.scalers[col].transform(df_targets[[col]]))
            else:
                # Should not happen if scalers are fitted correctly for all target cols
                processed_targets.append(df_targets[[col]].values)
        return np.hstack(processed_targets)

    def _infer_bow_direction(self, event_labels):
        bow_directions = []
        for label in event_labels:
            if 'a_bow' in label.lower(): # Assuming 'a_bow' for up-bow, case-insensitive
                bow_directions.append('up')
            elif 'd_bow' in label.lower(): # Assuming 'd_bow' for down-bow, case-insensitive
                bow_directions.append('down')
            else:
                bow_directions.append('none') # Or another appropriate default
        return pd.DataFrame(bow_directions, columns=['bow_direction'])

    def _infer_is_transition(self, event_flags, event_labels):
        is_transitions = []
        for flag, label in zip(event_flags, event_labels):
            # Check for specific 'TRANSITION' in label, or if event_flag indicates a special type
            if 'TRANSITION' in label.upper() or not (1 <= flag <= 8): 
                is_transitions.append(1)
            else:
                is_transitions.append(0)
        return pd.DataFrame(is_transitions, columns=['is_transition'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Neural Network Model ---
class BehavioralCloningModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(BehavioralCloningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# --- Main Training Script ---
def train_bc_model(csv_files, model_save_path="bc_policy.pth", scalers_save_path="bc_scalers.pkl", encoders_save_path="bc_encoders.pkl"):
    all_data = []
    for f_name in csv_files:
        if os.path.exists(f_name):
            df = pd.read_csv(f_name, low_memory=False) 
            all_data.append(df)
        else:
            print(f"Warning: File not found: {f_name}. Skipping.")

    if not all_data:
        print("No CSV data loaded. Exiting BC training.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # --- Preprocessing: Fit Scalers and Encoders (on the full_df) ---
    scalers = {}
    
    # Handle current_note_number for fitting scalers:
    full_df['current_note_number'] = pd.to_numeric(full_df['current_note_number'], errors='coerce')
    full_df['current_note_number'] = full_df['current_note_number'].fillna(0) # Fill NaN with 0 for scaling

    numerical_feature_cols = [col for col in INPUT_FEATURE_COLS if col not in ['current_string', 'event_label', 'event_flag']]
    for col in numerical_feature_cols:
        scaler = MinMaxScaler()
        full_df[col] = scaler.fit_transform(full_df[[col]]) 
        scalers[col] = scaler
    
    for col in TARGET_COLS: # Targets are also numerical, scale them
        scaler = MinMaxScaler()
        full_df[col] = scaler.fit_transform(full_df[[col]])
        scalers[col] = scaler

    one_hot_encoders = {}
    # current_string encoder
    encoder_cs = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    all_strings = ['A', 'D', 'G', 'C'] 
    encoder_cs.fit(np.array(all_strings).reshape(-1, 1))
    one_hot_encoders['current_string'] = encoder_cs

    # bow_direction encoder (up/down/none)
    encoder_bd = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    all_bow_directions = ['up', 'down', 'none'] 
    encoder_bd.fit(np.array(all_bow_directions).reshape(-1, 1))
    one_hot_encoders['bow_direction'] = encoder_bd
    
    # Save scalers and encoders
    with open(scalers_save_path, 'wb') as f:
        pickle.dump(scalers, f)
    with open(encoders_save_path, 'wb') as f:
        pickle.dump(one_hot_encoders, f)

    # Split data into training and validation sets
    train_df, val_df = train_test_split(full_df, test_size=TEST_SIZE, random_state=42)

    # Create datasets and dataloaders
    train_dataset = CelloTrajectoryDataset(train_df, INPUT_FEATURE_COLS, TARGET_COLS, scalers, one_hot_encoders)
    val_dataset = CelloTrajectoryDataset(val_df, INPUT_FEATURE_COLS, TARGET_COLS, scalers, one_hot_encoders)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Determine input and output dimensions
    dummy_input_df = train_df.head(1)[INPUT_FEATURE_COLS]
    dummy_dataset = CelloTrajectoryDataset(dummy_input_df, INPUT_FEATURE_COLS, TARGET_COLS, scalers, one_hot_encoders)
    input_dim = dummy_dataset.X.shape[1]
    output_dim = len(TARGET_COLS)

    # Initialize model, loss, and optimizer
    model = BehavioralCloningModel(input_dim, output_dim, HIDDEN_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"BC Model Input Dimension: {input_dim}")
    print(f"BC Model Output Dimension: {output_dim}")

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("Training complete!")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# --- Example Usage ---
if __name__ == '__main__':
    train_bc_model(CSV_FILES)

    # --- How to load and use the trained model ---
    # When you want to use this BC policy in your UR5eCelloTrajectoryEnv:
    # 1. Load the scalers and encoders:
    # import pickle
    # with open('bc_scalers.pkl', 'rb') as f:
    #     loaded_scalers = pickle.load(f)
    # with open('bc_encoders.pkl', 'rb') as f:
    #     loaded_encoders = pickle.load(f)
    
    # 2. You'll need to define a function to preprocess a single real-time observation
    #    This function should replicate the logic in CelloTrajectoryDataset._preprocess_features
    #    for a single row (dictionary) of raw observation data.
    #
    #    def _preprocess_single_observation(raw_obs_dict, loaded_scalers, loaded_encoders):
    #        single_obs_df = pd.DataFrame([raw_obs_dict])
    #        
    #        # Apply the same cleaning for 'current_note_number' as during training
    #        single_obs_df['current_note_number'] = pd.to_numeric(single_obs_df['current_note_number'], errors='coerce')
    #        single_obs_df['current_note_number'] = single_obs_df['current_note_number'].fillna(0)
    #
    #        processed_features = []
    #        numerical_cols = [col for col in INPUT_FEATURE_COLS if col not in ['current_string', 'event_label', 'event_flag']]
    #        for col in numerical_cols:
    #            processed_features.append(loaded_scalers[col].transform(single_obs_df[[col]]))
    #
    #        current_string_encoded = loaded_encoders['current_string'].transform(single_obs_df[['current_string']])
    #        processed_features.append(current_string_encoded) # No .toarray()
    #
    #        # Replicate bow direction inference for a single observation
    #        # You'll need to pass the actual event_label value from the current observation
    #        # A robust way would be to make _infer_bow_direction a static method or a separate function
    #        # For now, let's assume you pass the `event_label` correctly
    #        bow_direction_value = CelloTrajectoryDataset(pd.DataFrame(), [], [], {}, {})._infer_bow_direction(single_obs_df['event_label'])[0] # This won't work easily if not static
    #        # A better way for single obs:
    #        if 'a_bow' in str(raw_obs_dict.get('event_label', '')).lower(): bow_dir = 'up'
    #        elif 'd_bow' in str(raw_obs_dict.get('event_label', '')).lower(): bow_dir = 'down'
    #        else: bow_dir = 'none'
    #        bow_direction_encoded = loaded_encoders['bow_direction'].transform(np.array([[bow_dir]]))
    #        processed_features.append(bow_direction_encoded) # No .toarray()
    #
    #        # Replicate is_transition inference for a single observation
    #        # Same considerations for static method/separate function apply
    #        # For single obs:
    #        is_transition_val = 0
    #        if 'TRANSITION' in str(raw_obs_dict.get('event_label', '')).upper() or not (1 <= raw_obs_dict.get('event_flag', 0) <= 6):
    #            is_transition_val = 1
    #        processed_features.append(np.array([[is_transition_val]]))
    #
    #        return np.hstack(processed_features).flatten()
    #
    # 3. Load the model:
    # model_input_dim = # <--- Use the input dimension printed during training (e.g., 22)
    # loaded_model = BehavioralCloningModel(model_input_dim, len(TARGET_COLS), HIDDEN_SIZE)
    # loaded_model.load_state_dict(torch.load('bc_policy.pth'))
    # loaded_model.eval() # Set to evaluation mode
    #
    # 4. In your UR5eCelloTrajectoryEnv's step method (or where you generate actions):
    #    # Assuming `raw_obs_dict` is a dictionary of current robot state and musical context
    #    # Example:
    #    # raw_obs_dict = {
    #    #    'q_base': self.data.qpos[0], 'q_shoulder': self.data.qpos[1], ...,
    #    #    'TCP_pose_x': self.data.site_xpos[0][0], 'TCP_pose_y': self.data.site_xpos[0][1], ...,
    #    #    'time_elapsed_sec': self.data.time,
    #    #    'remaining_duration_sec': (current_note_end_time - self.data.time),
    #    #    'current_note_number': get_current_midi_note_number(), 
    #    #    'current_string': get_current_string_name(),
    #    #    'event_label': get_current_event_label(),
    #    #    'event_flag': get_current_event_flag()
    #    # }
    #
    #    preprocessed_obs_array = _preprocess_single_observation(raw_obs_dict, loaded_scalers, loaded_encoders)
    #    obs_tensor = torch.tensor(preprocessed_obs_array, dtype=torch.float32).unsqueeze(0) # Add batch dimension
    #    
    #    with torch.no_grad():
    #        bc_action_tensor_scaled = loaded_model(obs_tensor)
    #    
    #    # Inverse transform the predicted joint positions to their original scale
    #    bc_action_unscaled = np.zeros(len(TARGET_COLS))
    #    for i, col in enumerate(TARGET_COLS):
    #        val_scaled = bc_action_tensor_scaled.squeeze(0).numpy()[i]
    #        bc_action_unscaled[i] = loaded_scalers[col].inverse_transform(np.array([[val_scaled]]))[0][0]
    #    
    #    # This `bc_action_unscaled` is your `target_q_baseline` in original joint angle units
    #    # Then you would combine it with your RL residual: `total_target_q = bc_action_unscaled + rl_action_residual`
    #    # And send `total_target_q` to `_apply_pid_control`.