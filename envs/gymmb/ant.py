import numpy as np
import torch
from gym.envs.mujoco import AntEnv # use mujoco.ant_v3 to use Ant-v3 which allows to have current xy-pos and compute x_velocity from states
import torch

from envs.task import Task


class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        
        #Note: This has been set to work with  exclude_current_positions_from_observation=False as it is by default
        # However, to do so, need to include the x_velocity in the state (so that can compute rwd from states)
        # This is done by over-writing the envs methods to include the velocity (e.g. get_obs has one extra entry with the velocity) 
        
        #### ========= Extra (delete) ============
        #x_velocity = (next_states[:,0] - states[:,0]) / 0.05 # 0.05 refers to default dt value

        # I don't think need to check if it is healhty as this is done by the is_done method (i.e. check GYMMB_Walker2d implemnetation)
        #min_z, max_z = (0.2, 1.0) # Default values for healthy range
        #healthy_cond_1 = torch.isfinite(states).all(dim=1)
        #healthy_cond_2 = torch.logical_and(states[:,2] >= min_z, states[:,2] <= max_z) # Assuming current_poss included in obs
        #healthy_rwd = torch.maximum(0, torch.tensor(healthy_cond_1 + healthy_cond_2, device=states.device) - torch.ones(states.size()[0],device=states.device)) * 1 # Here *1 refers to the default healthy_reward value 

        # ==========================================
        forward_rwd = next_states[:,0] 

        healthy_rwd = 1
        control_cost = 0.5 * actions.pow(2).sum(dim=1)
        contact_forces = torch.clip(states[:,28:],-1.0,1.0) # Default values # Note: not 100% sure, the doc says there are 6 external forces for 14 links
        contact_cost = 5e-4 * contact_forces.pow(2).sum(dim=1)

        return  forward_rwd + healthy_rwd - control_cost - contact_cost

        



class GYMMB_Ant(AntEnv):
    def __init__(self):
        self.prev = None
        super().__init__() # use everything as default 
    
    def step(self,a):
        
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        if self.prev is not None:
            self.prev = self.get_additional_obs()

        return ob, None, None, {}

    def _get_obs(self):

        curr = self.get_additional_obs()
        delta = curr - self.prev if self.prev is not None else np.zeros_like(curr)

        return np.concatenate([
            delta / self.dt,
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext,-1,1).flat,
        ])

    def reset_model(self):

            self.set_state(
                    self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size = self.model.nq),
                    self.init_qvel + 0.1 * self.np_random.standard_normal(self.model.nv)
            )

            self.prev = self.get_additional_obs()
            return self._get_obs()

    def get_additional_obs(self):
            
            return np.array([self.sim.data.qpos[0]]) # only take x coord, since needed to compute velocity in x coord

    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod
    def is_done(states):

        notdone = (states[:,1] >= 0.2) & (states[:,1] <= 1.0) # Since at the first entry of the state the x_vel has been included and using default value healthy_z_range
        return ~notdone
