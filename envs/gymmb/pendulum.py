import torch
import numpy as np
from gym import Env
from gym.envs.classic_control import PendulumEnv

class GYMMB_Pendulum(PendulumEnv):

    def __init__(self):
        super().__init__()

    @staticmethod
    def is_done(states):
        bs = states.shape[0]
        return torch.zeros(size=[bs], dtype=torch.bool, device=states.device)  # Always False
