import numpy as np 

class Policy:
    '''
    Implemtation of an RL policy for determining an appropriate non-linear force-TCP function.
    '''

    def __init__(self, params):
        '''
        params: loaded from config.yaml.
        '''
        self.params = params
        
        # define force limits and initial force
        self.force_cfg = params['force']
        self._curr_force = self.force_cfg['init']
        self.max_force = self.force_cfg['max']
        self.min_force = self.force_cfg['min']
        self.max_delta = self.force_cfg['max_delta']

    def forward(self, state: np.ndarray):
        pass 

    def act(self, state: np.ndarray) -> np.ndarray:
        '''
        Given the current state, return a force action.
        state is a (12,) vector of concatenated [force_6dof, tcp_6dof].
        For now, force is kept constant, so state is ignored, but this will be changed.
        '''

        # clipping is defensive programming
        # we already handled this in our gp_gate, but just in case... :) 
        return np.clip(self._curr_force, self.min_force, self.max_force)
    
    def get_parameters(self):
        '''
        Return current policy parameters as a dict.
        For now, just the current force, but this will be expanded as we add more complexity to the policy.
        '''
        return {'force': self._curr_force}
    
    def set_parameters(self, params: dict):
        '''
        Update policy parameters from a dict.
        For now, just updates the current force, but this will be expanded as we add more complexity to the policy.
        '''
        new_force = params.get('force', self._curr_force)
        delta = new_force - self._curr_force

        # enforce max delta constraint
        if abs(delta) > self.max_delta:
            new_force = self._curr_force + np.sign(delta) * self.max_delta

        # clip to force limits (defensive programming)
        self._curr_force = np.clip(new_force, self.min_force, self.max_force)

