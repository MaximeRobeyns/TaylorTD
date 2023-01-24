from gym.envs.mujoco.humanoid import HumanoidEnv

class GYMMB_Humanoid(HumanoidEnv):
    def __init__(self):
        super().__init__()
    

    @staticmethod    
    def is_done(states):
        notdone = (states[:,0] > 1.0) & (states[:,0] < 2.0)
        return ~notdone
