mage:
	python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=5. td_error_weight=1.

dyna:
	python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=0. td_error_weight=1.

taylor:
	echo "Not Implemented!"

