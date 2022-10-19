import numpy as np
from gym.envs.mujoco.humanoid import HumanoidEnv
import torch

from envs.task import Task

# Needed to work with MAGE interface
class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        
        return None


class GYMMB_Humanoid(HumanoidEnv):
    def __init__(self):
        super().__init__()
    

# Needed to work with MAGE interface
    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod    
    def is_done(states):
        notdone = (states[:,0] > 1.0) & (states[:,0] < 2.0)
        return ~notdone
