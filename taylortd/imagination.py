import torch


class SingleStepImagination:
    def __init__(
        self,
        model,
        initial_states,
        n_actors,
        model_sampling_type,
        grad_state,
        det_action,
    ):
        """This is a less general but more efficient version of imagination.many_steps for n_steps=1"""

        self.initial_states = initial_states

        self.model = model
        self.n_actors = n_actors
        self.model_sampling_type = model_sampling_type
        self.device = model.device
        self.grad_state = grad_state

        self.det_action = det_action

    def reset(self):
        pass

    def many_steps(self, agent):
        idx = torch.randint(self.initial_states.shape[0], size=[self.n_actors])
        states = self.initial_states[idx].to(self.device)

        # NOTE: Add the state.require_grad before computing the actions so that the current action can be differentiate through the state
        if self.grad_state:
            states = (
                states.detach().requires_grad_()
            )  # create an alias of states which share data, but requires_grad, in this way we are not changing the grad flag of data stored in the buffer

        actions = agent.get_action(
            states, deterministic=self.det_action
        )  # here can explore over actions to learn Q  with deterministic=False or take the det_action with no exploration over action

        next_states = self.model.sample(
            states, actions, self.model_sampling_type
        )
        return states, actions, next_states
