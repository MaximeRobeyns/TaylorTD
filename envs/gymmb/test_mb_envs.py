import warnings
warnings.filterwarnings(action='ignore', category=Warning)

import unittest
import gym
import torch
# noinspection PyUnresolvedReferences
import envs.gymmb
from wrappers import IsDoneEnv

SEED = 123456


def to_torch(x):
    x = torch.from_numpy(x)
    if x.ndimension() == 1:
        x = x.unsqueeze(0)
    return x.double()


class TestEnvironments(unittest.TestCase):
    def any_test(self, standard_env_name, my_env_name,
                 state_orig_cmp_indices=slice(0, None), state_mine_cmp_indices=slice(0, None),
                 ignore_reward=False):
        # noinspection PyUnresolvedReferences
        env_orig = gym.make(standard_env_name)
        env_orig.seed(SEED)
        # In humanoid use the actual done condition from the env and not from the wrapper
        env_mine = IsDoneEnv(gym.make(my_env_name))

        env_mine.seed(SEED)
        sorig = env_orig.reset()
        smine = env_mine.reset()
        self.assertSequenceEqual(list(sorig[state_orig_cmp_indices]), list(smine[state_mine_cmp_indices]))
        for step_i in range(10500):
            action = env_mine.action_space.sample()
            sorig, r1, d1, _ = env_orig.step(action)
            s2_prev = smine.copy()
            smine, r2, d2, _ = env_mine.step(action)

            self.assertSequenceEqual(list(sorig[state_orig_cmp_indices]), list(smine[state_mine_cmp_indices]))
            self.assertEqual(d1, d2)
            if not ignore_reward:
                self.assertAlmostEqual(r1, r2, places=3)
            if d1:
                sorig = env_orig.reset()
                smine = env_mine.reset()
                self.assertSequenceEqual(list(sorig[state_orig_cmp_indices]), list(smine[state_mine_cmp_indices]))


    def test_cheetah(self):
        self.any_test('HalfCheetah-v2', 'GYMMB_HalfCheetah-v2')

    def test_walker2d(self):
        self.any_test('Walker2d-v2', 'GYMMB_Walker2d-v2')

    def test_pendulum(self):
        self.any_test('Pendulum-v0', 'GYMMB_Pendulum-v0')

    def test_humanoid(self):
        self.any_test('Humanoid-v2', 'GYMMB_Humanoid-v2' )
