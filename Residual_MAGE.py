import copy
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from ddpg import Actor
from models import get_activation
from radam import RAdam
import numpy as np

logger = logging.getLogger(__file__)

class ActionValueFunction(nn.Module):
    def __init__(self, d_state, d_action, n_layers, n_units, activation):
        super().__init__()
        assert n_layers >=1, "# of hidden layers"
        
        # Create the input layer with the correspeonding activation function
        layers = [nn.Linear(d_state + d_action, n_units), get_activation(activation)]
        
        # Add n hidden layers
        for lyr_idx in range(1, n_layers):
            layers += [nn.Linear(n_units, n_units), get_activation(activation)]

        # Add the output layer    
        layers += [nn.Linear(n_units, 1)]

        # Pass the layers to torch API, nn.Sequential
        self.layers = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.layers(x)


def inner_product_last_dim(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.unsqueeze(-2)@B.unsqueeze(-1)).squeeze(-1)

class Residual_MAGE(nn.Module):
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
            tdg_error_weight=0
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
        self.step_counter = 0
        self.tdg_error_weight = tdg_error_weight


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
        #noise = (
        #        torch.randn_like(actions) * self.policy_noise
        #).clamp(-self.noise_clip, self.noise_clip)
        # frozen next action vector

        raw_next_actions = self.actor(next_states) #self.actor_target(next_states)
        next_actions = raw_next_actions #(raw_next_actions + noise).clamp(-1, 1) # Note for Residual want next action to be correct one, not noisy version, and not Traget pol?
        
        # compute the target Q value depending on the update
        
        next_Q = self.critic(next_states, next_actions) # both next_s and next_a carry a gradient, which should relate them to action
        q_target = rewards.unsqueeze(1) + self.discount * masks.float().unsqueeze(1) * next_Q
        zero_targets = torch.zeros_like(q_target, device=self.device)

        # Get current Q estimates; multiple values due to TD3
        q = self.critic(states, actions)
        q_td_error = q_target - q

        # We apply Taylor Direct / Residual updates with both q1_td_error and q2_td_error.
        # q1_td_error and q2_td_error are O

        critic_loss, standard_loss, gradient_loss = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)


        if self.value_loss == 'huber':
            loss_fn = F.smooth_l1_loss
        elif self.value_loss == 'mse':
            loss_fn = F.mse_loss
        else:
            raise ValueError(f'Unexpected loss: "{self.value_loss}"')


        # Compute the True Residual update for MAGE: =================================
        
        standard_loss = 0.5 * loss_fn(q_td_error,zero_targets)

        if self.tdg_error_weight !=0:

                # Shape: [batch, actions]
                gradients_error_norms = torch.autograd.grad(outputs=q_td_error, inputs=actions,
                                           grad_outputs=torch.ones(q_td_error.size(), device=self.device),
                                           retain_graph=True, create_graph=True,
                                           only_inputs=True)[0].flatten(start_dim=1).norm(dim=1, keepdim=True)


                # Compute magnitude of gradient of TD relative to actions
                gradient_loss = torch.mean(gradients_error_norms) #0.5 * loss_fn(gradients_error_norms, zero_targets)  
        

       
        critic_loss = standard_loss + self.tdg_error_weight * gradient_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        if self.step_counter % 1 == 0: #% self.policy_delay == 0: # For residual try to update policy every iteration
            # Compute actor loss
            q = self.critic(states, self.actor(states))  # originally in TD3 we had here q1 only
            actor_loss = -q.mean()
            self.last_actor_loss = actor_loss.item()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Update the frozen target policy
            #for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            #    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return raw_next_actions[0, 0].item(), standard_loss.item(), self.tdg_error_weight * gradient_loss.item(), self.last_actor_loss

    @staticmethod
    def catastrophic_divergence(q_loss, pi_loss):
        return q_loss > 1e2 or (pi_loss is not None and abs(pi_loss) > 1e5)
