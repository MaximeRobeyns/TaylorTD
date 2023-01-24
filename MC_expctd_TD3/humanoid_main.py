#!/usr/bin/env python


# This main is to be used with Humanoid and not with the old envs due to a change in how the rwd is computed
# in all old envs the rwd is re-computed based on the task() function, here it relies on the actual observed rwd 
# from the environment
import logging
import warnings

warnings.filterwarnings(action='ignore', module='importlib', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', module='dotmap', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', module='tensorflow', category=FutureWarning)
warnings.filterwarnings(action='ignore', module='tensorboard', category=FutureWarning)
warnings.filterwarnings(action='ignore', module='gym', category=FutureWarning)

from time import perf_counter
from dotmap import DotMap
from functools import wraps

import numpy as np
import torch
import os
import sacred
import gym
from env_loop import EnvLoop
from datetime import datetime
from logger import configure_logger
from metriclogger import MetricLogger

# noinspection PyUnresolvedReferences
import envs  # Register custom gym envs
# noinspection PyUnresolvedReferences
import envs.gymmb  # Register custom gym envs
# noinspection PyUnresolvedReferences
import sacred_utils  # For a custom mongodb flag

from radam import RAdam
from reward_model import RewardModel
from td3 import TD3
from wrappers import BoundedActionsEnv, IsDoneEnv, MuJoCoCloseFixWrapper, RecordedEnv
from buffer import Buffer
from models import Model
from normalizer import TransitionNormalizer
from perturbed_imagination import SingleStepImagination
from utils import to_np, EpisodeStats


ex = sacred.Experiment(save_git_info=False)# Needed to add it to run on BP as don't have python git dependecies...

configure_logger()
logger = logging.getLogger(__file__)


""" Experiment Configuration """


# noinspection PyUnusedLocal
@ex.config
def main_config():
    n_total_steps = 100000                          # total number of steps in real environment (including warm up)
    n_warm_up_steps = 1000                          # number of steps on real MDP to populate the initial buffer, actions selected by random agent

    normalize_data = True                           # normalize states, actions, next states to zero mean and unit variance (both for model training and policy training)
    run_type = 'trial' 
    run_number = 88
    seed = 891 # REMOVE, for testing purpose only!

# noinspection PyUnusedLocal
@ex.config
def eval_config():
    eval_freq = 1000                                # interval in steps for testing in exploitations the trained models and agents, can be None
    n_eval_episodes_per_policy = 10                 # number of episodes each eval agent is evaluated for each task


# noinspection PyUnusedLocal
@ex.config
def env_config(n_total_steps):
    env_name = 'GYMMB_Humanoid-v2'               # environment name: GYMMB_* or Magellan*
    task_name = 'standard'                          # Name of task to perform within environment e.g. in half cheetah env. either 'running' or 'flipping'  


    env = gym.make(env_name)
    d_state = env.observation_space.shape[0]        # state dimensionality
    d_action = env.action_space.shape[0]            # action dimensionality
    del env


# noinspection PyUnusedLocal
@ex.config
def model_arch_config(env_name):
    model_ensemble_size = 8                         # number of models in the bootstrap ensemble
    model_n_units = 512                             # number of hidden units in each hidden layer (hidden layer size)
    model_n_layers = 4                              # number of hidden layers in the model (at least 2)
    model_activation = 'swish'                      # activation function (see models.py for options)

    if env_name == 'GYMMB_Humanoid-v2':
        reward_n_units = 512
    else:
        reward_n_units = 256


    reward_n_layers = 3
    train_reward = True                            # Whether to train the reward function (True) or use the hand-designed one (False)
    reward_activation = 'swish'


# noinspection PyUnusedLocal
@ex.config
def model_training_config():
    model_training_freq = 25                        # interval in steps between model trainings
    model_training_n_batches = 120                  # number of batches every model training (1200 corresponds to 15 epochs for 20k buffer)

    model_training_grad_clip = 5                    # gradient clipping to train model
    model_lr = 1e-4                                 # learning rate for training models
    model_weight_decay = 1e-4                       # L2 weight decay on model parameters (good: 1e-5, default: 0)
    model_batch_size = 256                          # batch size for training models

    model_sampling_type = 'ensemble'                # Procedure to use when sampling from ensemble of models, 'ensemble' or 'DS'


# TODO: For training/evaluation active/reactive consider using hierarchical dicts
# noinspection PyUnusedLocal
@ex.config
def policy_training_config(env_name):
    discount = 0.99                                # discount factor

    policy_training_freq = 1                       # interval (in real steps) between subsequent policy trainings
    policy_training_n_iters = 10                   # number of policy update iterations (img data + policy update)
    policy_training_n_updates_per_iter = 1         # number of on-policy optimizations steps

    policy_actors = 1024                           # number of parallel actors in imagination MDP


# noinspection PyUnusedLocal
@ex.config
def policy_arch_config(n_total_steps, env_name):

    if env_name == 'GYMMB_Humanoid-v2':
        policy_n_layers = 4
        value_n_layers = 4
    else:

        policy_n_layers = 2
        value_n_layers = 2


    # policy function
    policy_n_units = 400                            # number of units in each hidden layer
    policy_activation = 'swish'
    policy_lr = 1e-4                                # learning rate

    # value function
    value_n_units = 400                             # number of units in each hidden layer
    value_activation = 'swish'
    value_lr = 1e-4
    value_tau = 0.005                               # soft target network update mixing factor
    value_loss = 'huber'                            # 'huber' or 'mse'

    # common for value and policy
    agent_grad_clip = 5
    agent_alg = 'expected_td3'                               # td3 or ddpg

    # Parameters for TD3
    td3_policy_delay = 2
    td3_expl_noise = 0.1
    td3_action_cov = 0.25                            #in Taylor RL (covariance of action points) - 5 works really well (equivalent value to MAGE)
    td3_state_cov =0.00005
    det_action = True                         # Determines whether Q in model transitions evaluated for deterministic or stochastic policy
    

    data_buffer_size = n_total_steps          # Memory buffer size  



# noinspection PyUnusedLocal
@ex.config
def infra_config(env_name,agent_alg,run_type, td3_action_cov, td3_state_cov,  run_number):
    use_cuda = True                                 # if true use CUDA
    gpu_id = 0                                      # ID of GPU to use (by default use GPU 0)
    print_config = True                             # Set False if you don't want that (e.g. for regression tests)

    checkpoint = False # Use true to store checkpoints
    restart_checkpoint = False # Use to load model from an existing checkpoint
    checkpoint_freq = 25000

    if use_cuda and 'CUDA_VISIBLE_DEVICES' not in os.environ:  # gpu_id is used only if CUDA_VISIBLE_DEVICES was not set
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    
    update_dir =''

    if td3_action_cov > 0:
       update_dir = 'Action' 
    if td3_state_cov > 0:   
       update_dir = 'State' + update_dir
    if update_dir == '':
       update_dir = 'ZeroOrder'    

    self_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dir = '__default__'                        # Set dump_dir=None if you don't want to be create dump_dir
    if dump_dir == '__default__':
    #    dump_dir = os.path.join(self_dir, 'logs', f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{os.getpid()}')
        dump_dir = os.path.join(self_dir, 'results',f'{env_name}',f'{agent_alg}',f'{run_type}',update_dir,f'{run_number}') # Add this to save file in specif directory
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)

    omp_num_threads = 1                             # 1 is usually the correct choice, especially when using GPU



@ex.capture
def setup(seed, dump_dir, omp_num_threads, print_config, _run):
    """Sets random seeds and environment variables"""
    if print_config:
        ex.commands["print_config"]()
        print('Shell command:', sacred_utils.get_shell_command())
    if dump_dir is not None:
        sacred_utils.dump_shell_command(dump_dir)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(omp_num_threads)
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)


""" Initialization Helpers (Get) """


@ex.capture
def get_env(env_name): #, seed): # REMOVE seed from argument, only added from testing purposes

    """Setup the Gym environment"""
    env = gym.make(env_name)
    # clips actions before calling step.
    env = BoundedActionsEnv(env)

    # Adds done condition
    env = IsDoneEnv(env) # Still need done condition for imagination
    # Allows us to close mujoco better
    env = MuJoCoCloseFixWrapper(env)
       
    # REMOVE: Fixed all the seeds ------------

    env.seed(np.random.randint(np.iinfo(np.uint32).max))
    #env.seed(seed) # tried remove this to see if get variable accuracy

    if hasattr(env.action_space, 'seed'):  # Only for more recent gym
        env.action_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    if hasattr(env.observation_space, 'seed'):  # Only for more recent gym
        env.observation_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    return env

@ex.capture
def get_agent(mode, *, agent_alg):
    logger.debug(f"{ex.step_i:6d} | {mode} | getting fresh agent ...")

    return get_td3_agent()



@ex.capture
def get_td3_agent(*, d_state, d_action, discount, device, value_tau, value_loss, policy_lr,
                  value_lr, policy_n_units, value_n_units, policy_n_layers, value_n_layers, policy_activation,
                  value_activation, agent_grad_clip, td3_policy_delay, td3_expl_noise):
    return TD3(d_state=d_state, d_action=d_action, device=device, gamma=discount, tau=value_tau,
               value_loss=value_loss, policy_lr=policy_lr, value_lr=value_lr,
               policy_n_layers=policy_n_layers, value_n_layers=value_n_layers, value_n_units=value_n_units,
               policy_n_units=policy_n_units, policy_activation=policy_activation, value_activation=value_activation,
               grad_clip=agent_grad_clip, policy_delay=td3_policy_delay,expl_noise=td3_expl_noise)




@ex.capture
def get_random_agent(d_action, device):
    class RandomAgent:
        # noinspection PyUnusedLocal
        @staticmethod
        def get_action(states, deterministic=False):
            # This is not so nice since we hard-code [-1, +1] range but this is the only way to be compatible with the agent interface
            return torch.rand(size=(states.shape[0], d_action), device=device) * 2 - 1
    return RandomAgent()


def to_deterministic_agent(agent):
    class DeterministicAgent:
        @staticmethod
        def get_action(states):
            return agent.get_action(states, deterministic=True).detach()

    return DeterministicAgent()


@ex.capture
def get_model(d_action, d_state, model_ensemble_size, model_n_units, model_n_layers, model_activation, device, _run):
    logger.debug(f"{ex.step_i:6d} | getting fresh model ...")
    model = Model(d_action=d_action, d_state=d_state, ensemble_size=model_ensemble_size,
                  n_units=model_n_units, n_layers=model_n_layers, activation=model_activation,
                  device=device)
    return model


@ex.capture
def get_reward_model(d_action, d_state, reward_n_units, reward_n_layers, reward_activation, device, _run):
    logger.debug(f"{ex.step_i:6d} | getting fresh reward model ...")
    model = RewardModel(d_action=d_action, d_state=d_state,
                        n_units=reward_n_units, n_layers=reward_n_layers, activation=reward_activation,
                        device=device)
    return model


@ex.capture # Return the function (i.e. SingleStepImagination) to generate (imagined) one-step transitions based on the model of environment
def get_imagination(model, initial_states, *, model_sampling_type, policy_actors,td3_action_cov, td3_state_cov, det_action):
        return SingleStepImagination(model, initial_states, n_actors=policy_actors, model_sampling_type=model_sampling_type, action_cov=td3_action_cov, state_cov=td3_state_cov, det_action=det_action)


@ex.capture
def get_model_optimizer(params, *, model_lr, model_weight_decay):
    return RAdam(params, lr=model_lr, weight_decay=model_weight_decay)


@ex.capture
def get_reward_model_optimizer(params, *, model_lr, model_weight_decay):
    return RAdam(params, lr=model_lr, weight_decay=model_weight_decay)


@ex.capture
def get_buffer(d_state, d_action, n_total_steps, normalize_data, device, data_buffer_size):

    buffer = Buffer(d_action=d_action, d_state=d_state, size=data_buffer_size)
    if normalize_data:
        buffer.setup_normalizer(TransitionNormalizer(d_state, d_action, device))
    return buffer


""" Agent Training """

# I don't think this class is used; Mage, taylor and Dyna-TD3, all rely on transition provided by the model (imaginary) which are
# computed by the class below ImaginationTransitionsProvider, this class should be used for standard td3 methods, which rely on
# buffer transitions rather than imaginary ones
class BufferTransitionsProvider:
    def __init__(self, buffer, task, is_done, device, policy_actors):
        self.buffer = buffer
        self.task = task
        self.is_done = is_done
        self.device = device
        self.policy_actors = policy_actors

    def get_training_transitions(self, agent):
        states, actions, next_states, _ = self.buffer.view()
        idx = torch.randint(len(self.buffer), size=[self.policy_actors])
        states, actions, next_states = [x[idx].to(self.device) for x in [states, actions, next_states]]
        rewards = self.task(states, actions, next_states)
        dones = self.is_done(next_states)
        return states, actions, next_states, rewards, dones

# This class relies on input object imagination to generate an imaginary(predicted) transition (i.e. based on the model)
# it also relies on input task to compute the true reward give the transition and the task
class ImaginationTransitionsProvider:
    def __init__(self, imagination, task, is_done):
        self.imagination = imagination
        self.task = task
        self.is_done = is_done
        self.imagination.reset()

    def get_initial_step(self, agent):
        
        states,actions = self.imagination.initial_step(agent)

        return states, actions

    def get_training_transitions(self): 
        
        perturbed_states, perturbed_actions, next_states = self.imagination.perturbed_steps()
        dones = self.is_done(next_states) 
        return perturbed_states, perturbed_actions, next_states, dones


@ex.capture
def get_training_data_provider(model, buffer, is_done, task):
    initial_states, _, _, _ = buffer.view() # This returns all inital states so far in the buffer
    imagination = get_imagination(model, initial_states) # Creates an "imagination" obj needed to generate next states through the ImaginationTransitionsProvider 
    return ImaginationTransitionsProvider(imagination=imagination, task=task, is_done=is_done)


@ex.capture
def train_agent(agent, model, reward_model, buffer, task, task_name, is_done, mode, context_i, *, _run, device,
                policy_training_n_updates_per_iter, agent_alg, train_reward, policy_training_n_iters):
    """Policy optimisation step"""
    data_provider = get_training_data_provider(model, buffer, is_done, task)

    q_loss, pi_loss = np.nan, np.nan
    for img_step_i in range(1, policy_training_n_iters + 1):
         unperturbed_states, unperturbed_actions = data_provider.get_initial_step(agent)

         for img_update_i in range(1, policy_training_n_updates_per_iter + 1):

            if len(unperturbed_states) == 0:
               continue

            states, actions, next_states, dones = data_provider.get_training_transitions()
            rewards = reward_model(states, actions, next_states).squeeze(1)

       
            raw_action, q_loss, pi_loss = agent.update(unperturbed_states, states, actions, rewards, next_states, masks=~dones) # Key method call to the critic and actor update
            # This is rare but can still happen
            if agent.catastrophic_divergence(q_loss, pi_loss):
                logger.info("Catastrophic divergence detected. Agent reset.")
                agent = get_agent('train')
                agent.setup_normalizer(model.normalizer)

    mode_task = f'{mode} | t:{task_name}'
    logger.debug(f"{ex.step_i:6d} | {mode_task} | pi_loss: {pi_loss:6.3f}; q_loss: {q_loss:6.3f}")

    if 'rewards' in locals():
        log_dict = dict(pi_loss=pi_loss, q_loss=q_loss, 
                        img_rewards_avg=rewards.mean().item(),
                        img_rewards_max=rewards.max().item(),
                        raw_action=raw_action,
                        update_batch_size=states.size()[0])
        ex.mlog.add_scalars(f"{mode}/{task_name}/mb_final", log_dict, **context_i)

    return agent


""" Model Training """


@ex.capture
def model_train_epoch(model, buffer, optimizer, model_batch_size, model_training_grad_clip):
    losses = []  # stores loss after each minibatch gradient update
    for states, actions, state_deltas in buffer.train_batches(ensemble_size=model.ensemble_size, batch_size=model_batch_size):
        optimizer.zero_grad()
        loss = model.loss(states, actions, state_deltas)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=model_training_grad_clip)
        optimizer.step()

    return losses


@ex.capture
def train_model(model, optimizer, buffer, mode, model_training_n_batches, *, _run):
    logger.debug(f"{ex.step_i:6d} | {mode} | training model...")
    n_target_batches = model_training_n_batches

    loss = np.nan
    batch_i = 0
    while batch_i < n_target_batches:
        losses = model_train_epoch(model=model, buffer=buffer, optimizer=optimizer)
        batch_i += len(losses)
        loss = np.mean(losses)
        logger.log(5, f'{ex.step_i:6d} | {mode} | batch {batch_i:3d} | model training loss: {loss:.2f}')
        ex.mlog.add_scalar(f"{mode}/model/train_loss", loss, batch_i=batch_i)

    logger.debug(f"{ex.step_i:6d} | {mode} | model training final loss : {loss:.3f}")
    ex.mlog.add_scalar(f"{mode}/model/training_loss_final", loss)


@ex.capture
def reward_model_train_epoch(reward_model, buffer, optimizer, task, model_batch_size, model_training_grad_clip):
    losses = []  # stores loss after each minibatch gradient update
    for states, actions, rewards, state_deltas in buffer.train_batches_rwd(ensemble_size=1, batch_size=model_batch_size):
        next_states = states + state_deltas
        states, actions, rewards, next_states = states.squeeze(0), actions.squeeze(0), rewards.squeeze(), next_states.squeeze(0)
        #rewards = task(states, actions, next_states)
        optimizer.zero_grad()
        loss = reward_model.loss(states, actions, next_states, rewards)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(reward_model.parameters(), clip_value=model_training_grad_clip)
        optimizer.step()

    return losses


@ex.capture
def train_reward_model(reward_model, optimizer, buffer, mode, model_training_n_batches, task, *, _run):
    logger.debug(f"{ex.step_i:6d} | {mode} | training reward model...")
    n_target_batches = model_training_n_batches

    loss = np.nan
    batch_i = 0
    while batch_i < n_target_batches:
        losses = reward_model_train_epoch(reward_model=reward_model, buffer=buffer, task=task, optimizer=optimizer)
        batch_i += len(losses)
        loss = np.mean(losses)
        logger.log(5, f'{ex.step_i:6d} | {mode} | batch {batch_i:3d} | reward model training loss: {loss:.2f}')
        ex.mlog.add_scalar(f"{mode}/reward_model/train_loss", loss, batch_i=batch_i)

    logger.debug(f"{ex.step_i:6d} | {mode} | reward model training final loss : {loss:.3f}")
    ex.mlog.add_scalar(f"{mode}/reward_model/training_loss_final", loss)


""" Evaluation method (testing model/buffer on task) """


@ex.capture
def evaluate_on_task(agent, model, buffer, task, task_name, context, *,  _run,
                     n_eval_episodes_per_policy,  dump_dir):
    """ Evaluate agent or model & agent """
    episode_returns, episode_lengths = [], []

    env_loop = EnvLoop(get_env,run=_run)
    agent = to_deterministic_agent(agent) # This ensures evalutation performance are based on the deterministic (optimal) action 

    # Test agent on real environment by running an episode
    for ep_i in range(1, n_eval_episodes_per_policy + 1):

        with torch.no_grad():
            states, actions, rewards,next_states = env_loop.episode(agent)

        ep_return = rewards.sum().item()
        ep_len = len(rewards)
        logger.log(15, f"{ex.step_i:6d} | {context} | t:{task_name} | episode {ep_i} | score: {ep_return:5.2f}")
        ex.mlog.add_scalars(f'{context}/{task_name}/episode/', {'return': ep_return, 'length': ep_len},
                            ep_i=ep_i)
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
    env_loop.close()

    return episode_returns, episode_lengths


def evaluate_on_tasks(agent, model, buffer, task_name, context):
    logger.info(f"{ex.step_i:6d} | {context} | evaluating model for tasks...")
    env = get_env()
    task = env.unwrapped.tasks()[task_name]
    env.close()

    ep_returns, ep_lengths = evaluate_on_task(agent, model, buffer, task, task_name, context)
    avg_ep_return = np.mean(ep_returns)
    std_ep_return = np.std(ep_returns)
    avg_ep_length = np.mean(ep_lengths)

    logger.info(f"{ex.step_i:6d} | {context} | t:{task_name} | avg. return: {avg_ep_return:5.2f}+-{std_ep_return:5.2f}")
    ex.mlog.add_scalars(f'{context}/{task_name}/avg_episode/', {'return': avg_ep_return, 'length': avg_ep_length})

    return avg_ep_return


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        ex.mlog.add_scalar(f'duration/{func.__name__}', end - start)
        return result
    return wrapper


""" Main Methods (sacred commands) """


@ex.capture
def log_last_episode(stats, *, _run):
    for task_name, task in stats.tasks.items():
        last_ep_return = stats.ep_returns[task_name][-1]
        last_ep_len = stats.ep_lengths[task_name][-1]
        logger.info(f'{ex.step_i:6d} | train | t:{task_name} | return: {last_ep_return:5.1f} ({last_ep_len:3d} steps)')
        ex.mlog.add_scalars(f"train/{task_name}/episode", {'return': last_ep_return, 'length': last_ep_len})



class MainTrainingLoop:
    """ Resembles ray.Trainable """

    @ex.capture
    def __init__(self, *, task_name):
        logger.info(f"Executing training...")
        
        tmp_env = get_env()
        self.is_done = tmp_env.unwrapped.is_done
        self.eval_tasks = {task_name: tmp_env.tasks()[task_name]}
        # Below returns the StandardTask obj (i.e. reference to the dict returned by .tasks() with task_name='standard')
        # StandardTask obj has one __call__ method which computes the rwd, that's why self.exploitation_task computes the rwd
        self.exploitation_task = tmp_env.tasks()[task_name] 
        del tmp_env

        ex.step_i = 0
        # initialise the state-space forward model
        # Note that this uses an ensemble network to calculate uncertainty; we
        # could replace it with an epinet.
        self.model = get_model()
        self.reward_model = get_reward_model()
        # Uses the 'Rectified Adam' (arxiv.org/abs/1908.03265) optimiser
        self.model_optimizer = get_model_optimizer(self.model.parameters())
        self.reward_model_optimizer = get_reward_model_optimizer(self.reward_model.parameters())
        self.buffer = get_buffer()
        self.agent = get_agent(mode='train')
        # setup state, action, state_delta normalisation
        self.agent.setup_normalizer(self.buffer.normalizer)
        # computes rewards for each time step in the episode. when episode
        # ends, this logs the total return and episode length
        self.stats = EpisodeStats(self.eval_tasks)
        self.last_avg_eval_score = None
        ex.mlog = None

        # Not considered part of the state
        self.new_experiment = True
        self.random_agent = get_random_agent()

        self._common_setup()
        
        # Use to load a pre-trained model
        if self.restart_checkpoint and self.dump_dir is not None:
            model_dir = os.path.join(self.dump_dir,'CheckPoint','Model.pt')
            MODEL = torch.load(model_dir)
            self.buffer = MODEL['Memory_buffer']
            self.agent.actor.load_state_dict(MODEL['Agent'])
            self.agent.actor_target.load_state_dict(MODEL['Target_Agent'])
            self.agent.actor_optimizer.load_state_dict(MODEL['Agent_optim'])
            self.agent.critic.load_state_dict(MODEL['Critic'])
            self.agent.critic_target.load_state_dict(MODEL['Target_Critic'])
            self.agent.critic_optimizer.load_state_dict(MODEL['Critic_optim'])
            self.model.load_state_dict(MODEL['Env_model'])
            self.model_optimizer.load_state_dict(MODEL['Env_model_optim'])
            self.reward_model.load_state_dict(MODEL['Rwd_model'])
            self.reward_model_optimizer.load_state_dict(MODEL['Rwd_model_optim'])
    
    @ex.capture
    def _common_setup(self,*, dump_dir, checkpoint, restart_checkpoint, checkpoint_freq, _run):
        
        self.env_loop = EnvLoop(get_env, run=_run)
        self.checkpoint = checkpoint
        self.dump_dir = dump_dir
        self.restart_checkpoint = restart_checkpoint
        self.checkpoint_freq = checkpoint_freq
    
    @ex.capture
    def _setup_if_new(self):

        if self.new_experiment:
            self.new_experiment = False
            ex.mlog = MetricLogger(ex)

    @ex.capture
    def train(self, *, device, n_total_steps, n_warm_up_steps, 
              model_training_freq, policy_training_freq, eval_freq,
              task_name, model_training_n_batches, train_reward):

        self._setup_if_new()

        ex.step_i += 1

        behavioral_agent = self.random_agent if ex.step_i <= n_warm_up_steps else self.agent
        with torch.no_grad():
                action = behavioral_agent.get_action(self.env_loop.state, deterministic=False).detach().to('cpu') # KEY: ensure real transition are sampled based on stochastic policy
        # save s
        prev_state = self.env_loop.state.clone().to(device)


        # take a step, s, s' p()
        state, reward, next_state, done = self.env_loop.step(to_np(action))

        # add transition to the buffer; (s, a, s', r)
        self.buffer.add(state, action, next_state, torch.from_numpy(np.array([[reward]], dtype=np.float)))
        self.stats.add(state, action, reward, next_state, done, compute_rwd=False)
        if done:
            log_last_episode(self.stats)

        tasks_rewards = {f'{task_name}': self.stats.get_recent_reward(task_name) for task_name in self.eval_tasks}
        step_stats = dict(
            step=ex.step_i,
            done=done,
            action_abs_mean=action.abs().mean().item(),
            reward= reward,
            action_value=self.agent.get_action_value(prev_state, action).item(),
        )
        ex.mlog.add_scalars('main_loop', {**step_stats, **tasks_rewards})

        # (Re)train the model on the current buffer
        if model_training_freq is not None and model_training_n_batches > 0 and ex.step_i % model_training_freq == 0:
            # setup normalisation for actor and critic
            self.model.setup_normalizer(self.buffer.normalizer)
            self.reward_model.setup_normalizer(self.buffer.normalizer)
            #
            # train_model is the main event.
            #
            timed(train_model)(self.model, self.model_optimizer, self.buffer, mode='train')
            if train_reward:
                task = self.exploitation_task
                timed(train_reward_model)(self.reward_model, self.reward_model_optimizer, self.buffer, mode='train', task=task)

        # (Re)train the policy using current buffer and model
        if ex.step_i >= n_warm_up_steps and ex.step_i % policy_training_freq == 0:
            task = self.exploitation_task
            self.agent.setup_normalizer(self.buffer.normalizer)
            #
            # train_agent is the main event.
            #
            self.agent = timed(train_agent)(self.agent, self.model, self.reward_model, self.buffer, task=task, task_name=task_name, is_done=self.is_done,
                                            mode='train', context_i={})

        # Evaluate the agent
        if eval_freq is not None and ex.step_i % eval_freq == 0:
            self.last_avg_eval_score = evaluate_on_tasks(agent=self.agent, model=self.model, buffer=self.buffer, task_name=task_name, context='eval')

        experiment_finished = ex.step_i >= n_total_steps
        
        if ex.step_i % self.checkpoint_freq == 0  and self.checkpoint and self.dump_dir is not None:

            model_dir = os.path.join(self.dump_dir,'CheckPoint')
            os.makedirs(model_dir, exist_ok=True)
            
            torch.save({
                #'N_step': ex.step_i,
                #'Memory_buffer': self.buffer,
                'Normaliser': self.buffer.normalizer,
                'Agent': self.agent.actor.state_dict(),
                'Target_Agent': self.agent.actor_target.state_dict(),
                #'Agent_optim': self.agent.actor_optimizer.state_dict(),
                #'Critic': self.agent.critic.state_dict(),
                #'Target_Critic': self.agent.critic_target.state_dict(),
                #'Critic_optim': self.agent.critic_optimizer.state_dict(),
                #'Env_model' : self.model.state_dict(),
                #'Env_model_optim': self.model_optimizer.state_dict(),
                #'Train_rwd': train_reward,
                #'Rwd_model': self.reward_model.state_dict(),
                #'Rwd_model_optim': self.reward_model_optimizer.state_dict(),
            }, os.path.join(model_dir,'Model.pt'))


        return DotMap(
            done=experiment_finished,
            avg_eval_score=self.last_avg_eval_score,
            action_abs_mean=action.abs().mean().item(),  # This is just for regression tests
            step_i=ex.step_i)

    def stop(self):
        self.env_loop.close()
        if ex.mlog is not None:
            ex.mlog.save_artifacts()


@ex.automain
def train():
    
    # Ensure cuda is available
    assert torch.cuda.is_available(), "No GPU"
    
    # main entrypoint.
    setup()
    training = MainTrainingLoop()

    # dot-access dict subclass.
    res = DotMap(done=False)
    while not res.done:
        res = training.train()

    training.stop()

    return res.get('avg_eval_score'), res.get('action_abs_mean')
