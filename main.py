'''
Initializes all training parameters, models, and calls training when done.
'''

import numpy as np 
import yaml 
import pickle
from training.train import Train
from models.gp_gate import GPGate
from models.policy import Policy
from reward.reward import Reward

# ------------- LOADING DATA AND MODELS -------------
def load_config(path='config.yaml'):
    '''Loads the config file for training, returns dictionary of parameters'''
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise Exception(f"Config file not found at {path}. Please check the path and try again.")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")

def load_gp_gate():
    '''Loads both the GP gate and its associated warmup data'''
    try:
        with open('gp_gate.pkl', 'rb') as f:
            gp_gate = pickle.load(f)
        return gp_gate
    except FileNotFoundError:
        print("GP gate file not found. Will initialize in training loop.")
        return None 

def load_gp_warmup():
    '''Loads the warmup data for the GP gate'''
    try:
        with open('warmup_data.npy', 'rb') as f:
            warmup_data = np.load(f)
        return warmup_data
    except FileNotFoundError:
        raise Exception("Warmup data file not found. Ensure 'warmup_data.npy' is in the correct directory.")
    except Exception as e:  
        raise Exception(f"Error loading warmup data: {e}")
    
    
# ------------- CALL TRAINING LOGIC -------------
if __name__ == "__main__":
    # initialize config, gp date, policy, & reward
    config = load_config()
    warmup_data = load_gp_warmup()

    gp_gate = load_gp_gate()
    if gp_gate is None:
        uncertainty_threshold = config["gp_uncertainty_threshold"]
        gp_gate = GPGate(warmup_data, uncertainty_threshold=uncertainty_threshold)

    policy = Policy(config["policy_params"])
    reward = Reward()

    # initialize training class
    trainer = Train(
        policy=policy,
        gp=gp_gate,
        reward_calc=reward,
        warmup_data=warmup_data,
        config=config
    )
    
    # start training loop
    trainer.train()
