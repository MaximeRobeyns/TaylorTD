import torch
from gym.envs.mujoco import HalfCheetahEnv


class GYMMB_HalfCheetah(HalfCheetahEnv):
    def __init_(self):
        super().__init__()

    @staticmethod
    def is_done(states):
        bs = states.shape[0]
        return torch.zeros(size=[bs], dtype=torch.bool, device=states.device)  # Always False
