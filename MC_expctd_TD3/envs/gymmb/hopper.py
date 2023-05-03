from gym.envs.mujoco import HopperEnv
import torch

class GYMMB_Hopper(HopperEnv):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def is_done(states):
        # A small difference: I check whether clipped states are finite instead of raw states
        # Also: states[0] is `delta qpos[0] / dt` instead of qpos[0]
        notdone = torch.all(torch.isfinite(states), dim=1) & (states[:, 1:].abs() < 100).all(dim=1) & (states[:, 0] > .7) & (states[:, 1].abs() < .2)
        return ~notdone
