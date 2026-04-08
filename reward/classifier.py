import numpy as np

class SoundClassifier:
    '''
    Sound classification model that takes in audio data and outputs a score.
    This score is then transformed into a reward signal by the Reward class.
    '''
    def __init__(self):
        pass 

    def predict(self) -> float:
        '''
        Given raw audio data, output a score representing the quality of the bowing.
        For now, this is just a placeholder that returns a random score between 0 and 1.
        '''
        return np.random.rand()
    
