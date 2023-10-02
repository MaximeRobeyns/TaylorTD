import torch
import logging

from collections import defaultdict

log = logging.getLogger("max.utils")


def to_torch(x):
    x = torch.from_numpy(x).float()
    if x.ndimension() == 1:
        x = x.unsqueeze(0)
    return x


def to_np(x):
    x = x.detach().cpu().numpy()
    if len(x.shape) >= 1:
        x = x.squeeze(0)
    return x


class EpisodeStats:
    """
    Computes rewards fore at each time step in the episode. When episode ends (done==True) it
    logs the total return and episode length

    Args:
        tasks: a list of tasks
    """

    def __init__(self, task_name):
        self.task_name = task_name
        self.curr_episode_rewards = defaultdict(list)
        self.ep_returns = defaultdict(list)
        self.ep_lengths = defaultdict(list)
        self.last_reward = defaultdict(float)

    def add(self, state, action, rwd, next_state, done):

        step_reward = rwd
        self.curr_episode_rewards[self.task_name].append(step_reward)
        self.last_reward[self.task_name] = step_reward
        if done:
            self.ep_returns[self.task_name].append(
                sum(self.curr_episode_rewards[self.task_name])
            )
            self.ep_lengths[self.task_name].append(
                len(self.curr_episode_rewards[self.task_name])
            )
            self.curr_episode_rewards[self.task_name].clear()

    def get_recent_reward(self):
        return self.last_reward[self.task_name]
