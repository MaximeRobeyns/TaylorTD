import torch
import gym
from ddpg import Actor
from utils import to_np, to_torch
from normalizer import TransitionNormalizer

env = gym.make('Humanoid-v2')

agent = Actor(env.observation_space.shape[0], env.action_space.shape[0], 4, 400, 'swish')

dic = torch.load('/user/work/px19783/code_repository/RL_project/ExpectedTD/results/GYMMB_Humanoid-v2/expected_td3/trial/StateAction/88/CheckPoint/Model.pt',map_location='cpu')
normalizer = dic['Normaliser']
agent.load_state_dict(dic['Agent'])

done = False
tot_rwd = []
i = 0

state = env.reset()
while not done:


    state = normalizer.normalize_states(to_torch(state))
    action = agent(state).detach()

    next_state, rwd, done, _ = env.step(to_np(action))


    tot_rwd.append(rwd)
    state = next_state

    print('Iteraction ',i, ' Accuracy: ',sum(tot_rwd))
    i+=1
    print(done)

print('Accuracy: ', sum(tot_rwd))
