import torch


class SingleStepImagination:

    def __init__(self, model, initial_states, n_actors, model_sampling_type,action_cov, state_cov, det_action):
        """ This is a less general but more efficient version of imagination.many_steps for n_steps=1 """

        self.initial_states = initial_states

        self.model = model
        self.n_actors = n_actors
        self.model_sampling_type = model_sampling_type
        self.device = model.device
        self.action_cov = action_cov
        self.state_cov = state_cov

        self.det_action = det_action

    def reset(self):
        pass

    def many_steps(self, agent):
        idx = torch.randint(self.initial_states.shape[0], size=[self.n_actors])
        states = self.initial_states[idx].to(self.device)

        perturbed_states = states + torch.randn_like(states) * self.state_cov

        actions = agent.get_action(perturbed_states, deterministic= self.det_action).detach() # here can explore over actions to learn Q  with deterministic=False or take the det_action with no exploration over action

        perturbed_actions = actions + torch.randn_like(actions) * self.action_cov 
        next_states = self.model.sample(perturbed_states, perturbed_actions, self.model_sampling_type)
        return states, perturbed_states, perturbed_actions, next_states # Also return unpertubed_states for actor update
