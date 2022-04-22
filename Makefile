taylor:
	python main.py with env_name=GYMMB_Pendulum-v0 agent_alg=td3_taylor td3_action_cov=0.1 td3_update_type='direct' td3_update_order=1 render=True n_warm_up_steps=1000

mage:
	python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=5. td_error_weight=1. render=True

test:
	python main.py with env_name=GYMMB_Pendulum-v0 agent_alg=td3 tdg_error_weight=5. td_error_weight=1. render=True

dyna:
	python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=0. td_error_weight=1.

