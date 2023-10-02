import gym
from gym.envs.mujoco import PusherEnv


class GYMMB_Pusher(gym.Wrapper):

    def __init__(self):
        super().__init__(PusherEnv())

    @staticmethod
    def is_done(states):
        done = torch.zeros((len(states),), dtype=torch.bool,
                           device=states.device)
        return done
