import numpy as np
from gym.envs.mujoco import HumanoidEnv
import torch

from envs.task import Task

# Note: Humanoid only works with train reward
class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        
        return None
        



class GYMMB_Humanoid(HumanoidEnv):
    def __init__(self):
        super().__init__()
    

    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod
    def is_done(states):

        notdone = (states[:,2] >= 1.0) & (states[:,2] <= 2.0) # This only works if exclude_current_positions_from_observation=False
        return ~notdone
