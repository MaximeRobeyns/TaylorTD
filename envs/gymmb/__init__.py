from gym.envs.registration import register

register(
    id='GYMMB_Ant-v2',
    entry_point='envs.gymmb.ant:GYMMB_Ant',
    max_episode_steps=1000
)

register(
    id='GYMMB_Humanoid-v2',
    entry_point='envs.gymmb.humanoid:GYMMB_Humanoid',
    max_episode_steps=1000
)


register(
    id='GYMMB_HalfCheetah-v2',
    entry_point='envs.gymmb.cheetah:GYMMB_HalfCheetah',
    max_episode_steps=1000
)

register(
    id='GYMMB_Pusher-v2',
    entry_point='envs.gymmb.pusher:GYMMB_Pusher',
    max_episode_steps=100,
    reward_threshold=0.0,
)


register(
    id='GYMMB_Walker2d-v2',
    entry_point='envs.gymmb.walker2d:GYMMB_Walker2d',
    max_episode_steps=1000
)


register(
    id='GYMMB_Pendulum-v0',
    entry_point='envs.gymmb.pendulum:GYMMB_Pendulum',
    max_episode_steps=200
)


register(
    id='GYMMB_Test-v0',
    entry_point='envs.gymmb.cheetah:GYMMB_HalfCheetah',
    max_episode_steps=100
)

