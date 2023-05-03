from gym.envs.mujoco import AntEnv
import torch

class GYMMB_Ant(AntEnv):
    def __init__(self):
        super().__init__()
    

    @staticmethod    
    def is_done(states):
        notdone = torch.all(torch.isfinite(states), dim=1) & (states[:,0] >= 0.2) & (states[:,0] <= 1.0) # defualt values
        return ~notdone
