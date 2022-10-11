import numpy as np
from gym.envs.mujoco import AntEnv
import torch

from envs.task import Task


class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        
        #Note: This only works if exclude_current_positions_from_observation=False, since need to access x_pos to compute velocity from states

        x_velocity = (next_states[:,0] - states[:,0]) ./ 0.05 # 0.05 refers to default dt value
        min_z, max_z = (0.2, 1.0) # Default values for healthy range
        healthy_cond_1 = np.isfinite(states).all(axis=1)
        healthy_cond_2 = np.logical_and(states[:,2] >= min_z, states[:,2] <= max_z) # Assuming current_poss included in obs
        healthy_rwd = np.maximum(0, healthy_cond_1 + healthy_cond_2 -1) * 1 # Here *1 refers to the default healthy_reward value 
        control_cost = 0.5 * np.sum(np.square(action),axis=1)

        return  x_velocity + healthy_rwd - control_cost

        



class GYMMB_Ant(AntEnv):
    def __init__(self):
        super().__init__(ctrl_cost_weight=0.5, use_contact_forces=False,healthy_z_range=(0.2, 1.0),exclude_current_positions_from_observation=False)

    
    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod
    def is_done(states):
        bs = states.shape[0]
        return torch.zeros(size=[bs], dtype=torch.bool, device=states.device)  # Always False
