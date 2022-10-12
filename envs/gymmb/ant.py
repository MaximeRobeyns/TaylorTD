import numpy as np
import torch
from gym.envs.mujoco import AntEnv # use mujoco.ant_v3 to use Ant-v3 which allows to have current xy-pos and compute x_velocity from states
import torch

from envs.task import Task


class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        
        #Note: This has been set to work with  exclude_current_positions_from_observation=False as it is by default
        # However, to do so, need to use z_pos at next state instead of x_velocity as the forward_rwd as done for GYMMB_Walker2d
        
        #### ========= Extra (delete) ============
        #x_velocity = (next_states[:,0] - states[:,0]) / 0.05 # 0.05 refers to default dt value

        # I don't think need to check if it is healhty as this is done by the is_done method (i.e. check GYMMB_Walker2d implemnetation)
        #min_z, max_z = (0.2, 1.0) # Default values for healthy range
        #healthy_cond_1 = torch.isfinite(states).all(dim=1)
        #healthy_cond_2 = torch.logical_and(states[:,2] >= min_z, states[:,2] <= max_z) # Assuming current_poss included in obs
        #healthy_rwd = torch.maximum(0, torch.tensor(healthy_cond_1 + healthy_cond_2, device=states.device) - torch.ones(states.size()[0],device=states.device)) * 1 # Here *1 refers to the default healthy_reward value 

        # ==========================================
        delta_pos = next_states[:,0] # Here used same strategy as it was used for Walker2d use pos at next state instead of velocity  

        forward_rwd = delta_pos 
        healthy_rwd = 1
        control_cost = 0.5 * actions.pow(2).sum(dim=1)
        contact_forces = torch.clip(states[:,27:],-1.0,1.0) # Default values # Note: not 100% sure, the doc says there are 6 external forces for 14 links
        contact_cost = 5e-4 * contact_forces.pow(2).sum(dim=1)

        return  forward_rwd + healthy_rwd - control_cost - contact_cost

        



class GYMMB_Ant(AntEnv):
    def __init__(self):
        super().__init__() # use everything as default 
    
    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod
    def is_done(states):

        notdone = (states[:,0] >= 0.2) & (states[:,0] <= 1.0) # Assuming current_poss excluded in obs (i.e. default) and using default value healthy_z_range
        return ~notdone
