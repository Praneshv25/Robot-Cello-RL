'''
Initializes all training parameters, models, and calls training when done.
'''

import numpy as np
import yaml
import pickle
from training.train import Train
from models.gp_gate import GPGate


def load_config(path='config.yaml'):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise Exception(f"Config file not found at {path}.")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")


def load_gp_gate():
    try:
        with open('gp_gate.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def load_gp_warmup():
    try:
        with open('warmup_data.npy', 'rb') as f:
            return np.load(f)
    except FileNotFoundError:
        raise Exception("warmup_data.npy not found.")
    except Exception as e:
        raise Exception(f"Error loading warmup data: {e}")


if __name__ == "__main__":
    config = load_config()

    warmup_data = load_gp_warmup()
    gp_gate = load_gp_gate()
    if gp_gate is None:
        gp_gate = GPGate(warmup_data, uncertainty_threshold=config["gp_uncertainty_threshold"])

    trainer = Train(gp=gp_gate, config=config)
    trainer.train()
