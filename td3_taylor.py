import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpg import Actor
from models import ParallelLinear, get_activation
from radam import RAdam
import numpy as np


class ActionValueFunction(nn.Module):
    def __init__(self, d_state, d_action, n_layers, n_units, activation):
        super().__init__()

        if n_layers == 0:
            # Use linear q function
            self.layers = ParallelLinear(d_state + d_action, 1, ensemble_size=2)
        else:
            layers = [ParallelLinear(d_state + d_action, n_units, ensemble_size=2), get_activation(activation)]
            for lyr_idx in range(1, n_layers):
                layers += [ParallelLinear(n_units, n_units, ensemble_size=2), get_activation(activation)]
            layers += [ParallelLinear(n_units, 1, ensemble_size=2)]
            self.layers = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = x.unsqueeze(0).repeat(2, 1, 1)
        return self.layers(x)


class TD3_Taylor(nn.Module):
    """

    TODO: actually update this class to implement the taylor rl udpate step.

    """
    def __init__(
            self,
            d_state,
            d_action,
            device,
            gamma,
            tau,
            policy_lr,
            value_lr,
            value_loss,
            value_n_layers,
            value_n_units,
            value_activation,
            policy_n_layers,
            policy_n_units,
            policy_activation,
            grad_clip,
            policy_delay=2,
            policy_noise=0.2,
            noise_clip=0.5,
            expl_noise=0.1,
            action_cov=0.1,
            td3_update = 'residual'
    ):
        super().__init__()

        # Create the actor network. This is a simple MLP.
        # A forward pass maps states to a vector of actions, bound to [-1, 1] by tanh
        self.actor = Actor(d_state, d_action, policy_n_layers, policy_n_units, policy_activation).to(device)
        # Copy of the actor with frozen weights.
        self.actor_target = copy.deepcopy(self.actor)
        # Optimisation algorithm for the actor.
        self.actor_optimizer = RAdam(self.actor.parameters(), lr=policy_lr)

        # Create the critic; similarly to the actor, this is just an MLP.
        self.critic = ActionValueFunction(d_state, d_action, value_n_layers, value_n_units, value_activation).to(device)
        # Frozen version of the critic network
        self.critic_target = copy.deepcopy(self.critic)
        # Critic activation function
        self.critic_optimizer = RAdam(self.critic.parameters(), lr=value_lr)

        self.discount = gamma
        # soft target network update mixing factor
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        # exploration noise / dithering strategy
        self.expl_noise = expl_noise
        # normalizer pre-processes states
        self.normalizer = None
        self.value_loss = value_loss
        # prevents catestrophic weight movements
        self.grad_clip = grad_clip
        self.device = device
        self.last_actor_loss = 0

        self.action_cov = action_cov
        self.step_counter = 0

    def setup_normalizer(self, normalizer):
        self.normalizer = copy.deepcopy(normalizer)

    def get_action(self, states, deterministic=False):
        """Get action gets the action vector; a vector of elements bound to the
        [-1, 1] range. Can be interpreted as logits."""
        states = states.to(self.device)
        with torch.no_grad():
            if self.normalizer is not None:
                states = self.normalizer.normalize_states(states)
            actions = self.actor(states)
            if not deterministic:
                actions += torch.randn_like(actions) * self.expl_noise
            return actions.clamp(-1, +1)

    def get_action_with_logp(self, states):
        """Returns action vector, with gradient info???"""
        states = states.to(self.device)
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
        a = self.actor(states)
        return a, torch.ones(a.shape[0], device=a.device) * np.inf  # inf: should not be used

    def get_action_value(self, states, actions):
        """Returns Q(s, a) by calling the critic."""
        with torch.no_grad():
            states = states.to(self.device)
            actions = actions.to(self.device)
            return self.critic(states, actions)[0]  # just q1

    def update(self, states, actions, logps, rewards, next_states, masks):
        """This seems to be the main event, where the policy is updated"""
        if self.normalizer is not None:
            # normalise / pre-process state information, both this state and
            # next state.
            states = self.normalizer.normalize_states(states)
            next_states = self.normalizer.normalize_states(next_states)
        self.step_counter += 1

        ## Noise-corrupted action vector:
        # Select action according to policy and add clipped noise
        # This is the standard TD3 action-selection procedure
        noise = (
                torch.randn_like(actions) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)
        # frozen next action vector
        raw_next_actions = self.actor_target(next_states)
        next_actions = (raw_next_actions + noise).clamp(-1, 1)

        # Compute the target Q value
        next_Q1, next_Q2 = self.critic_target(next_states, next_actions)
        next_Q = torch.min(next_Q1, next_Q2)
        q_target = rewards.unsqueeze(1) + self.discount * masks.float().unsqueeze(1) * next_Q
        zero_targets = torch.zeros_like(q_target, device=self.device)

        # Get current Q estimates; multiple values due to TD3
        q1, q2 = self.critic(states, actions)
        q1_td_error, q2_td_error = q_target - q1, q_target - q2

        # We apply Taylor Direct / Residual updates with both q1_td_error and q2_td_error.
        # q1_td_error and q2_td_error are O

        critic_loss, td_loss, ag_loss = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        
        # Residual Vs Direct update ==========================================================
        
        # 1st order residual updates ------------------------------------
        if td3_update == 'residual':  

            # 1. square the td error
            q1_td_err_2 = (q1_td_error.unsqueeze(-2) @ q1_td_error.unsqueeze(-1)).squeeze(-1)
            q2_td_err_2 = (q2_td_error.unsqueeze(-2) @ q2_td_error.unsqueeze(-1)).squeeze(-1)

            
            # Compute standard critic loss
            if self.value_loss == 'huber':
                td_loss = 0.5 * (F.smooth_l1_loss(q1_td_err_2, zero_targets) + F.smooth_l1_loss(q2_td_err_2, zero_targets))
            elif self.value_loss == 'mse':
                td_loss = 0.5 * (F.mse_loss(q1_td_err_2, zero_targets) + F.mse_loss(q2_td_err_2, zero_targets))


            # Compute gradient critic loss
            dac1 = torch.autograd.grad(outputs=q1_td_error, inputs=actions,
                                                         grad_outputs=torch.ones(q1_td_error.size(), device=self.device),
                                                         retain_graph=True, create_graph=True,
                                                         only_inputs=True)[0].flatten(start_dim=1).norm(dim=1, keepdim=True)

            dac2 = torch.autograd.grad(outputs=q2_td_error, inputs=actions,
                                                         grad_outputs=torch.ones(q2_td_error.size(), device=self.device),
                                                         retain_graph=True, create_graph=True,
                                                         only_inputs=True)[0].flatten(start_dim=1).norm(dim=1, keepdim=True)

            dac1_2 = (dac1.unsqueeze(-2) @ dac1.unsqueeze(-1)).squeeze(-1)
            dac2_2 = (dac2.unsqueeze(-2) @ dac2.unsqueeze(-1)).squeeze(-1)

            if self.value_loss == 'huber':
                ag_loss = 0.5 * (F.smooth_l1_loss(dac1_2, zero_targets) + F.smooth_l1_loss(dac2_2, zero_targets))
            elif self.value_loss == 'mse':
                ag_loss = 0.5 * (F.mse_loss(dac1_2, zero_targets) + F.mse_loss(dac2_2, zero_targets))
        
        # 1st order direct update -------------------------------------
        else: 
            
            # Compute standard critic loss
            if self.value_loss == 'huber':
                td_loss = 0.5 * (F.smooth_l1_loss(q1_td_error, zero_targets) + F.smooth_l1_loss(q2_td_error, zero_targets))
            elif self.value_loss == 'mse':
                td_loss = 0.5 * (F.mse_loss(q1_td_error, zero_targets) + F.mse_loss(q2_td_error, zero_targets))
            
            # Compute gradient of TD-error with respect to action
            dac1 = torch.autograd.grad(outputs=q1_td_error, inputs=actions,
                                                         grad_outputs=torch.ones(q1_td_error.size(), device=self.device),
                                                         retain_graph=True, create_graph=True,
                                                         only_inputs=True)[0].flatten(start_dim=1)#.norm(dim=1, keepdim=True)
           
            dac2 = torch.autograd.grad(outputs=q2_td_error, inputs=actions,
                                                         grad_outputs=torch.ones(q2_td_error.size(), device=self.device),
                                                         retain_graph=True, create_graph=True,
                                                         only_inputs=True)[0].flatten(start_dim=1)#.norm(dim=1, keepdim=True)



            # Compute gradient of Q() with respect to action
            dQa1 = torch.autograd.grad(outputs=q1, inputs=actions,
                                                         grad_outputs=torch.ones(q1.size(), device=self.device),
                                                         retain_graph=True, create_graph=True,
                                                         only_inputs=True)[0].flatten(start_dim=1)#.norm(dim=1, keepdim=True)
            
            dQa2 = torch.autograd.grad(outputs=q2, inputs=actions,
                                                         grad_outputs=torch.ones(q2.size(), device=self.device),
                                                         retain_graph=True, create_graph=True,
                                                         only_inputs=True)[0].flatten(start_dim=1)#.norm(dim=1, keepdim=True)
            
            ag_loss = dac1.unsqueeze(-2).detach() @ dQa1 + dac2.unsqueeze(-2).detach() @ dQa2


            





        critic_loss = td_loss + self.action_cov * ag_loss

        # TODO add options to compute 2nd order residual updates, as well as
        # 1st and 2nd order direct updates.

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        if self.step_counter % self.policy_delay == 0:
            # Compute actor loss
            q1, q2 = self.critic(states, self.actor(states))  # originally in TD3 we had here q1 only
            q_min = torch.min(q1, q2)
            actor_loss = -q_min.mean()
            self.last_actor_loss = actor_loss.item()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Update the frozen target policy
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update the frozen target value function
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return raw_next_actions[0, 0].item(), td_loss.item(), self.action_cov * ag_loss.item(), self.last_actor_loss

    @staticmethod
    def catastrophic_divergence(q_loss, pi_loss):
        return False
        # return q_loss > 1e2 or (pi_loss is not None and abs(pi_loss) > 1e5)
