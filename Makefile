test:
	python main.py with env_name=GYMMB_Pendulum-v0 agent_alg=td3_taylor td3_action_cov=5 td3_update_type=action td3_update_order=1 render=False

mage:  ## mage
	python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=5. td_error_weight=1. render=True

kernel:  ## To setup a Jupyter kernel to run notebooks in SPItorch virtual env
	python -m ipykernel install --user --name taylor_rl \
		--display-name "Taylor RL (Python 3.9)"

lab: ## To start a Jupyter Lab server
	jupyter lab --notebook-dir=notebooks --ip=0.0.0.0 --port 8882 # --collaborative --no-browser

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
