import numpy as np
from reward.classifier import SoundClassifier
from pathlib import Path


class Reward:
    '''
    Implemtationo of reward function for RL model.
    Takes score from sound classification model and transforms it into a 
    reward signal.
    '''
    
    def __init__(self):
        self.rewards = []
        # load in CNN classification model - dummy for now
        self.classifier = SoundClassifier()

    def compute_reward(self, audio_file):
        """Offline reward from a saved .wav file."""
        ext = Path(audio_file).suffix
        assert ext == ".wav", f"audio_file must be a .wav file, got {ext}"
        self.audio_file = audio_file

        # get score from sound classification model
        score = self.classifier.predict(audio_file)
        reward = self.calculate_reward_from_score(score)
        self.rewards.append(reward)
        return reward

    @staticmethod
    def calculate_reward_from_score(score: float) -> float:
        """There needs to be a change in reward score"""
        return 2.0 * score - 1.0

