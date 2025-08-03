# Isaac Lab Command Shortcuts

ISAAC_LAB_PATH = ..\..\IsaacLab
ISAAC_LAB_CMD = $(ISAAC_LAB_PATH)\isaaclab.bat -p

# Default task
DEFAULT_TASK = t1

# Default target to show usage
help:
	@echo Available targets:
	@echo   isaac ARGS="command"  - Run command with IsaacLab prefix
	@echo   Example: make isaac ARGS="scripts/skrl/train.py --task Isaac-T1-v0 --headless"
	@echo   debug [task]          - Run debug training script with 1 env (default: ${DEFAULT_TASK})
	@echo   train [task]          - Train environment with SKRL (default: ${DEFAULT_TASK})
	@echo   play [task]           - Play environment (default: ${DEFAULT_TASK})
	@echo   view-logs             - View training logs in TensorBoard
	@echo   view-recent-log       - View most recent training log in TensorBoard

# Run command with IsaacLab prefix
isaac:
	$(ISAAC_LAB_CMD) $(ARGS)

.PHONY: help isaac debug train play view-logs view-recent-log

debug:
	$(ISAAC_LAB_CMD) scripts/skrl/train.py --task $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),$(DEFAULT_TASK)) --num_envs 1

train:
	$(ISAAC_LAB_CMD) scripts/skrl/train.py --task $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),$(DEFAULT_TASK)) --headless

play:
	$(ISAAC_LAB_CMD) scripts/skrl/play.py --task $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),$(DEFAULT_TASK)) --num_envs 6

# Dummy target to prevent make from trying to build files named after tasks
%:
	@:

view-logs:
	@echo [INFO]if you run into error make sure to 'conda deactivate' so that it uses isaacs python
	$(ISAAC_LAB_CMD) -m tensorboard.main --logdir logs/skrl/booster_lab

view-recent-log:
	@echo [INFO] If you run into errors, make sure to 'conda deactivate' so that it uses Isaac\'s Python
	@for /f %%i in ('dir /b /ad /o-d logs\skrl\booster_lab') do ( \
		echo [INFO] Launching TensorBoard with logdir=logs\skrl\booster_lab\%%i && \
		$(ISAAC_LAB_CMD) -m tensorboard.main --logdir logs/skrl/booster_lab/%%i && \
		goto :eof \
	)

