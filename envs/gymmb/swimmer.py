from gym.envs.mujoco import SwimmerEnv
import torch


class GYMMB_Swimmer(SwimmerEnv):
    def __init__(self):
        self.prev = None
        super().__init__()

    @staticmethod
    def is_done(states):
        bs = states.shape[0]
        return torch.zeros(size=[bs], dtype=torch.bool, device=states.device)  # Always False
