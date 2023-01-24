import torch
import numpy as np
from gym.envs.classic_control import PendulumEnv

from envs.task import Task


# Needed for MAGE interface
class StandardTask(Task):

    def __call__(self, states, actions, next_states):
        
        return None

class GYMMB_Pendulum(PendulumEnv):

    def __init__(self):
        super().__init__()


    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod
    def is_done(states):
        bs = states.shape[0]
        return torch.zeros(size=[bs], dtype=torch.bool, device=states.device)  # Always False
