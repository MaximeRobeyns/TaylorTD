CMD 		= python main.py with
FLAGS 		= render=true env_name=GYMMB_Pendulum-v0 # GYMMB_HalfCheetah-v2

# arguments: 1: action_cov, 2: lr, 3: update_type, 4: order
td3_conf 	= $(shell echo agent_alg=td3_taylor td3_action_cov=$(1) value_lr=$(2) td3_update_type=\'$(3)\' td3_update_order=$(4))
ddpg_conf 	= $(shell echo agent_alg=ddpg_taylor ddpg_action_cov=$(1) value_lr=$(2) ddpg_update_type=\'$(3)\' ddpg_update_order=$(4))

TD3_FLAGS 	= agent_alg=td3_taylor td3_action_cov=5.00 value_lr=1e-3 td3_update_type='residual' td3_update_order=1
DDPG_FLAGS 	= agent_alg=ddpg_taylor ddpg_action_cov=4.00 ddpg_update_order=2 ddpg_update_type='residual'

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: 1r_td3 2r_td3 1d_td3 2d_td3

# TD3 -------------------------------------------------------------------------

1r_td3:  ## 1st order residual update td3
	$(CMD) $(FLAGS) $(call td3_conf,5.00,1e-4,residual,1)

2r_td3:  ## 2st order residual update td3
	$(CMD) $(FLAGS) $(call td3_conf,5.00,1e-4,residual,2)

1d_td3:  ## 1st order direct update td3
	$(CMD) $(FLAGS) $(call td3_conf,5.00,1e-4,direct,1)

2d_td3:  ## 2st order direct update td3
	$(CMD) $(FLAGS) $(call td3_conf,5.00,1e-4,direct,2)

# DDPG ------------------------------------------------------------------------

.PHONY: 1r_ddpg 2r_ddpg 1d_ddpg 2d_ddpg

1r_ddpg:  ## 1st order residual update ddpg
	$(CMD) $(FLAGS) $(call ddpg_conf,5.00,1e-4,residual,1)

2r_ddpg:  ## 2st order residual update ddpg
	$(CMD) $(FLAGS) $(call ddpg_conf,5.00,1e-4,residual,2)

1d_ddpg:  ## 1st order direct update ddpg
	$(CMD) $(FLAGS) $(call ddpg_conf,5.00,1e-4,direct,1)

2d_ddpg:  ## 2st order direct update ddpg
	$(CMD) $(FLAGS) $(call ddpg_conf,5.00,1e-4,direct,2)

# legacy

td3_mage:  ## td3 mage
	python main.py with env_name=GYMMB_Pendulum-v0 agent_alg=td3 tdg_error_weight=5. td_error_weight=1. render=True

mage:  ## mage
	python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=5. td_error_weight=1. render=True

dyna:  ## dyna
	python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=0. td_error_weight=1.


kernel:  ## To setup a Jupyter kernel to run notebooks in SPItorch virtual env
	python -m ipykernel install --user --name taylor_rl \
		--display-name "Taylor RL (Python 3.9)"


lab: ## To start a Jupyter Lab server
	jupyter lab --notebook-dir=. --ip=0.0.0.0  --port 8882 # --collaborative --no-browser
