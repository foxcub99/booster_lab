# Isaac Lab Command Shortcuts

ISAAC_LAB_PATH = ..\..\IsaacLab
ISAAC_LAB_CMD = $(ISAAC_LAB_PATH)\isaaclab.bat -p

# Default target to show usage
help:
	@echo Available targets:
	@echo   isaac ARGS="command"  - Run command with IsaacLab prefix
	@echo   Example: make isaac ARGS="scripts/skrl/train.py --task Isaac-T1-v0 --headless"
	@echo   debug                 - Run debug training script with 1 env
	@echo   train                 - Train T1 environment with SKRL
	@echo   play                  - Play T1 environment
	@echo   view-logs             - View training logs in TensorBoard

# Run command with IsaacLab prefix
isaac:
	$(ISAAC_LAB_CMD) $(ARGS)

.PHONY: help isaac debug train-t1 play-t1 train-skrl

debug:
	$(ISAAC_LAB_CMD) scripts/skrl/train.py --task Isaac-T1-v0 --num_envs 1

train:
	$(ISAAC_LAB_CMD) scripts/skrl/train.py --task Isaac-T1-v0 --headless

play:
	$(ISAAC_LAB_CMD) scripts/skrl/play.py --task Isaac-T1-v0 --num_envs 6

view-logs:
	@echo [INFO]if you run into error make sure to 'conda deactivate' so that it uses isaacs python
	$(ISAAC_LAB_CMD) -m tensorboard.main --logdir logs/skrl/t1_direct