from gym.envs.mujoco import Walker2dEnv


class GYMMB_Walker2d(Walker2dEnv):
    def __init__(self):
        super().__init__()

    @staticmethod
    def is_done(states):
        notdone = (states[:, 0] > 0.8) & (states[:, 0] < 2.) & (states[:, 1] > -1) & (states[:, 1] < 1)
        return ~notdone
