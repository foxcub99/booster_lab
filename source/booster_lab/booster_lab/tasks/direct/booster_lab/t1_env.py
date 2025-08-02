"""
T1 humanoid locomotion environment for Isaac Lab.

This environment is based on the original Isaac Gym T1 implementation and adapted for Isaac Lab.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.terrains import TerrainImporter
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from .t1_env_cfg import T1EnvCfg


class T1Env(DirectRLEnv):
    """Environment for T1 humanoid locomotion."""

    cfg: T1EnvCfg

    def __init__(self, cfg: T1EnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the T1 environment."""
        super().__init__(cfg, render_mode, **kwargs)

        # Joint and body indices # gym didn't actuate anything else
        legs_indices = self.robot.actuators["legs"].joint_indices
        self.dof_indices = torch.cat([legs_indices])
        self.leg_dof_indices = legs_indices

        self.contact_body_indices = None

        # Velocity filtering (matching Isaac Gym)
        self.filtered_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.filtered_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # for Root Acceleration calculation
        self.last_filtered_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_filtered_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # for foot slippage
        self.feet_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.last_feet_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)

        # Gait tracking (matching Isaac Gym)
        self.gait_frequency = (
            torch.ones(self.num_envs, device=self.device) * 1.5
        )  # Default 1.5 Hz
        self.gait_process = torch.zeros(self.num_envs, device=self.device)

        # Command tracking
        self.commands = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        # Gravity vector
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(
            self.num_envs, 1
        )
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)

        # Previous states for acceleration calculation - will be resized after robot initialization
        self.last_dof_vel = None
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.last_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )

        # Default joint positions (matching Isaac Gym)
        self.default_dof_pos = torch.zeros(self.num_envs, 12, device=self.device)
        # Set defaults based on initial configuration (Hip_Pitch, Knee_Pitch, Ankle_Pitch)
        self.default_dof_pos[:, 0] = -0.2  # Left_Hip_Pitch
        self.default_dof_pos[:, 3] = 0.4  # Left_Knee_Pitch
        self.default_dof_pos[:, 4] = -0.25  # Left_Ankle_Pitch
        self.default_dof_pos[:, 6] = -0.2  # Right_Hip_Pitch
        self.default_dof_pos[:, 9] = 0.4  # Right_Knee_Pitch
        self.default_dof_pos[:, 10] = -0.25  # Right_Ankle_Pitch

        # Episode tracking
        self.episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "survival",
                "tracking_lin_vel_x",
                "tracking_lin_vel_y",
                "tracking_ang_vel",
                "base_height",
                "orientation",
                "torques",
                "dof_vel",
                "dof_acc",
                "action_rate",
            ]
        }

    def _get_joint_indices(self):
        """Get joint indices for easier access."""
        # Check if robot is properly initialized
        if (
            not hasattr(self.robot, "_root_physx_view")
            or self.robot._root_physx_view is None
        ):
            print(
                "Warning: Robot PhysX view not yet initialized, deferring joint indexing"
            )
            return

        # body indices
        # self.body_indices = []
        # for name in self.cfg.bodies:
        #     try:
        #         # Use find_bodies method which returns (indices, names)
        #         idx, _ = self.robot.find_bodies(name)
        #         if len(idx) > 0:
        #             self.body_indices.append(idx[0])
        #         else:
        #             print(f"Warning: Body {name} not found in robot")
        #     except Exception as e:
        #         print(f"Warning: Error finding body {name}: {e}")

        # self.dof_indices = torch.tensor(
        #     self.robot.data.joint_names, dtype=torch.long, device=self.device
        # )

        # Contact body indices for feet
        self.contact_body_indices = []
        for name in self.cfg.contact_bodies:
            try:
                # Use find_bodies method which returns (indices, names)
                idx, _ = self.robot.find_bodies(name)
                if len(idx) > 0:
                    self.contact_body_indices.append(idx[0])
                else:
                    print(f"Warning: Body {name} not found in robot")
            except Exception as e:
                print(f"Warning: Error finding body {name}: {e}")

        self.contact_body_indices = torch.tensor(
            self.contact_body_indices, dtype=torch.long, device=self.device
        )

        # Initialize last_dof_vel with correct size now that we know the number of DOFs
        if self.last_dof_vel is None:
            num_dofs = self.robot.data.joint_vel.shape[-1]
            self.last_dof_vel = torch.zeros(self.num_envs, num_dofs, device=self.device)

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by the inverse of quaternion q."""
        q_w = q[:, 0]
        q_vec = q[:, 1:]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
        return a - b + c

    def _setup_scene(self):
        """Setup the scene with robot and terrain."""
        # Add robot
        self.robot = Articulation(self.cfg.robot)

        # Add terrain
        self.terrain = TerrainImporter(self.cfg.terrain)

        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Add collision sensors
        self.collision_sensors = ContactSensor(self.cfg.collision_sensors)
        self.scene.sensors["collision_sensors"] = self.collision_sensors

        self.foot_contact_sensors = ContactSensor(self.cfg.foot_contact_sensors)
        self.scene.sensors["foot_contact_sensors"] = self.foot_contact_sensors

    def reset(self, seed=None, options=None):
        """Override reset to initialize joint indices after first reset."""
        # Initialize joint indices before any reset logic if not already done
        if self.contact_body_indices is None:
            self._get_joint_indices()

        # Call parent reset
        result = super().reset(seed=seed, options=options)

        return result

    def _post_init_setup(self):
        """Setup joint and body indices after scene creation."""
        # This method is no longer needed as we use the reset override
        pass

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        # Clip actions like Isaac Gym
        self.actions = torch.clip(
            actions, -self.cfg.clip_actions, self.cfg.clip_actions
        )

        # Update gait process (like Isaac Gym)
        self.gait_process += self.gait_frequency * self.step_dt
        self.gait_process = torch.fmod(self.gait_process, 1.0)  # Keep in [0, 1] range

    def _apply_action(self) -> None:
        """Apply the processed actions to the robot - match Isaac Gym exactly."""
        # Calculate joint targets: default_pos + action_scale * actions (like Isaac Gym)
        if self.dof_indices is not None and len(self.dof_indices) > 0:
            dof_targets = self.default_dof_pos + self.cfg.action_scale * self.actions
            self.robot.set_joint_position_target(
                dof_targets, joint_ids=self.dof_indices
            )

    def _get_observations(self) -> dict:
        """Compute observations for the environment - match Isaac Gym exactly."""
        # Update filtered velocities (exponential moving average)
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b

        self.last_filtered_lin_vel = self.filtered_lin_vel.clone()
        self.filtered_lin_vel[:] = (
            base_lin_vel * self.cfg.filter_weight
            + self.filtered_lin_vel * (1.0 - self.cfg.filter_weight)
        )
        self.last_filtered_ang_vel = self.filtered_ang_vel.clone()
        self.filtered_ang_vel[:] = (
            base_ang_vel * self.cfg.filter_weight
            + self.filtered_ang_vel * (1.0 - self.cfg.filter_weight)
        )

        # Projected gravity (gravity in robot frame)
        base_quat = self.robot.data.root_quat_w
        self.projected_gravity[:] = self._quat_rotate_inverse(
            base_quat, self.gravity_vec
        )

        # Update feet positions
        self.last_feet_pos = self.feet_pos.clone()
        self.feet_pos = self.robot.data.body_state_w[:, self.contact_body_indices, :3]

        # Joint states
        joint_pos = self.robot.data.joint_pos[:, self.dof_indices]
        joint_vel = self.robot.data.joint_vel[:, self.dof_indices]
        default_joint_pos = self.default_dof_pos

        # Gait information (2D)
        gait_cos = torch.cos(2 * torch.pi * self.gait_process).unsqueeze(-1)
        gait_sin = torch.sin(2 * torch.pi * self.gait_process).unsqueeze(-1)

        # Commands scaling
        commands_scale = torch.tensor(
            [
                self.cfg.lin_vel_normalization,
                self.cfg.lin_vel_normalization,
                self.cfg.ang_vel_normalization,
            ],
            device=self.device,
        )

        # Isaac Gym observation structure: 47 elements
        # 3 (projected_gravity) + 3 (base_ang_vel) + 3 (commands) + 2 (gait) + 12 (joint_pos) + 12 (joint_vel) + 12 (actions) = 47
        obs = torch.cat(
            [
                self.projected_gravity * self.cfg.gravity_normalization,  # 3
                base_ang_vel * self.cfg.ang_vel_normalization,  # 3
                self.commands[:, :3] * commands_scale,  # 3
                gait_cos,  # 1
                gait_sin,  # 1
                (joint_pos - default_joint_pos) * self.cfg.dof_pos_normalization,  # 12
                joint_vel * self.cfg.dof_vel_normalization,  # 12
                self.actions,  # 12
            ],
            dim=-1,
        )
        # Total: 3 + 3 + 3 + 1 + 1 + 12 + 12 + 12 = 47 ✓

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward for the current step - match Isaac Gym implementation."""
        total_reward = torch.zeros(self.num_envs, device=self.device)

        survival_rew = self.cfg.survival_reward_scale
        total_reward += survival_rew

        lin_vel_x_error = torch.square(
            self.commands[:, 0] - self.filtered_lin_vel[:, 0]
        )
        lin_vel_x_rew = (
            torch.exp(-lin_vel_x_error / self.cfg.tracking_sigma)
            * self.cfg.tracking_lin_vel_x_reward_scale
        )
        total_reward += lin_vel_x_rew

        lin_vel_y_error = torch.square(
            self.commands[:, 1] - self.filtered_lin_vel[:, 1]
        )
        lin_vel_y_rew = (
            torch.exp(-lin_vel_y_error / self.cfg.tracking_sigma)
            * self.cfg.tracking_lin_vel_y_reward_scale
        )
        total_reward += lin_vel_y_rew

        ang_vel_error = torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2])
        ang_vel_rew = (
            torch.exp(-ang_vel_error / self.cfg.tracking_sigma)
            * self.cfg.tracking_ang_vel_reward_scale
        )
        total_reward += ang_vel_rew

        base_height = self.robot.data.root_pos_w[:, 2]
        height_error = torch.square(base_height - self.cfg.base_height_target)
        height_penalty = height_error * self.cfg.base_height_reward_scale
        total_reward += height_penalty

        collision_body_violations = (
            torch.norm(self.collision_sensors.data.net_forces_w, dim=-1) > 1.0
        )
        num_collision_violations = collision_body_violations.sum(dim=-1)
        collision_penalty = num_collision_violations * self.cfg.collision_reward_scale
        total_reward += collision_penalty

        lin_vel_z_penalty = (
            torch.square(self.filtered_lin_vel[:, 2]) * self.cfg.lin_vel_z_reward_scale
        )
        total_reward += lin_vel_z_penalty

        ang_vel_xy_penalty = (
            torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=-1)
            * self.cfg.ang_vel_xy_reward_scale
        )
        total_reward += ang_vel_xy_penalty

        orientation_penalty = (
            torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
            * self.cfg.orientation_reward_scale
        )
        total_reward += orientation_penalty

        # TODO check applied torque is equivalent to self.torques
        torque_penalty = (
            torch.sum(torch.square(self.robot.data.applied_torque), dim=-1)
            * self.cfg.torques_reward_scale
        )
        total_reward += torque_penalty

        dof_vel = self.robot.data.joint_vel
        dof_vel_penalty = (
            torch.sum(torch.square(dof_vel), dim=-1) * self.cfg.dof_vel_reward_scale
        )
        total_reward += dof_vel_penalty

        joint_acc = (self.last_dof_vel - dof_vel) / self.step_dt
        dof_acc_penalty = (
            torch.sum(torch.square(joint_acc), dim=-1) * self.cfg.dof_acc_reward_scale
        )
        total_reward += dof_acc_penalty

        root_lin_acc = (
            self.last_filtered_lin_vel - self.filtered_lin_vel
        ) / self.step_dt
        root_ang_acc = (
            self.last_filtered_ang_vel - self.filtered_ang_vel
        ) / self.step_dt
        root_acc_penalty = (
            torch.sum(torch.square(root_lin_acc), dim=-1)
            + torch.sum(torch.square(root_ang_acc), dim=-1)
        ) * self.cfg.root_acc_reward_scale
        total_reward += root_acc_penalty

        action_rate_penalty = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=-1)
            * self.cfg.action_rate_reward_scale
        )
        total_reward += action_rate_penalty

        # DOF position limits penalty - fixed shape handling
        joint_pos_limits_lower = self.robot.data.joint_pos_limits[
            0, :, 0
        ]  # [num_joints]
        joint_pos_limits_upper = self.robot.data.joint_pos_limits[
            0, :, 1
        ]  # [num_joints]

        lower = joint_pos_limits_lower + 0.5 * (1 - self.cfg.soft_dof_pos_limit) * (
            joint_pos_limits_upper - joint_pos_limits_lower
        )

        upper = joint_pos_limits_upper - 0.5 * (1 - self.cfg.soft_dof_pos_limit) * (
            joint_pos_limits_upper - joint_pos_limits_lower
        )

        dof_pos_limits_penalty = (
            torch.sum(
                (
                    (self.robot.data.joint_pos < lower)
                    | (self.robot.data.joint_pos > upper)
                ).float(),
                dim=-1,
            )
            * self.cfg.dof_pos_limits_reward_scale
        )
        total_reward += dof_pos_limits_penalty

        # lower_soft = lower_limits + 0.5 * (1 - self.cfg.soft_dof_pos_limit) * (
        #     upper_limits - lower_limits
        # )
        # upper_soft = upper_limits - 0.5 * (1 - self.cfg.soft_dof_pos_limit) * (
        #     upper_limits - lower_limits
        # )

        # dof_pos_limits_penalty = (
        #     torch.sum(
        #         ((joint_pos < lower_soft) | (joint_pos > upper_soft)).float(),
        #         dim=-1,
        #     )
        #     * self.cfg.dof_pos_limits_reward_scale
        # )

        # DOF velocity limits penalty TODO
        # if (
        #     hasattr(self.cfg, "dof_vel_limits_reward_scale")
        #     and self.cfg.dof_vel_limits_reward_scale != 0
        # ):
        #     vel_limit = 30.0  # From actuator config
        #     dof_vel_limits_penalty = (
        #         torch.sum(
        #             (
        #                 torch.abs(dof_vel) - vel_limit * self.cfg.soft_dof_vel_limit
        #             ).clamp(min=0.0, max=1.0),
        #             dim=-1,
        #         )
        #         * self.cfg.dof_vel_limits_reward_scale
        #     )
        #     total_reward += dof_vel_limits_penalty

        # Torque limits penalty TODO check same as above torque
        # if (
        #     hasattr(self.cfg, "torque_limits_reward_scale")
        #     and self.cfg.torque_limits_reward_scale != 0
        # ):
        #     torque_limit = 400.0  # From actuator config
        #     torque_limits_penalty = (
        #         torch.sum(
        #             (
        #                 torch.abs(self.robot.data.applied_torque)
        #                 - torque_limit * self.cfg.soft_torque_limit
        #             ).clamp(min=0.0),
        #             dim=-1,
        #         )
        #         * self.cfg.torque_limits_reward_scale
        #     )
        #     total_reward += torque_limits_penalty

        # Torque tiredness penalty TODO
        # there is no way to inherently get torque limits in isaac lab
        # i could implement a difference between calculated and applied but it's a small scale
        # total_reward += torque_tiredness_penalty

        power_penalty = torch.sum(
            (self.robot.data.applied_torque * self.robot.data.joint_vel).clip(min=0.0),
            dim=-1,
        )
        total_reward += power_penalty

        # missed feet slip
        feet_movement = torch.square(
            torch.sum(((self.last_feet_pos - self.feet_pos) / self.step_dt), dim=-1)
        )
        feet_ground_contact = self.foot_contact_sensors.data.net_forces_w[:, :, 2] > 0.0
        feet_slip = torch.sum(feet_movement * feet_ground_contact, dim=-1)
        feet_slip_penalty = (
            feet_slip
            * self.cfg.feet_slip_reward_scale
            * (self.episode_length_buf > 1).float()
        )
        total_reward += feet_slip_penalty

        # missed feet vel z
        # (disabled)

        # missed feet roll

        # missed feet yaw diff

        # missed feet yaw mean

        # missed reward feet distance

        # Simple swing reward based on gait phase
        left_swing = (
            torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg.swing_period
        ) & (self.gait_frequency > 1e-8)
        right_swing = (
            torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg.swing_period
        ) & (self.gait_frequency > 1e-8)

        # For now, assume feet are not in contact during swing (simplified)
        # TODO this needs a contact something
        feet_swing_reward = (
            left_swing.float() + right_swing.float()
        ) * self.cfg.feet_swing_reward_scale
        total_reward += feet_swing_reward

        # Store episode sums for tracking
        self.episode_sums["survival"] += survival_rew
        self.episode_sums["tracking_lin_vel_x"] += lin_vel_x_rew
        self.episode_sums["tracking_lin_vel_y"] += lin_vel_y_rew
        self.episode_sums["tracking_ang_vel"] += ang_vel_rew
        self.episode_sums["base_height"] += -height_penalty
        self.episode_sums["orientation"] += -orientation_penalty
        self.episode_sums["torques"] += -torque_penalty
        self.episode_sums["dof_vel"] += -dof_vel_penalty
        self.episode_sums["dof_acc"] += -dof_acc_penalty
        self.episode_sums["action_rate"] += -action_rate_penalty

        # Update previous states
        self.last_dof_vel = dof_vel.clone()
        self.last_root_vel = torch.cat(
            [self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w], dim=1
        ).clone()
        self.last_actions = self.actions.clone()

        return total_reward * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute done conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check if robot has fallen (base too low or tilted too much)
        base_height = self.robot.data.root_pos_w[:, 2]
        base_quat = self.robot.data.root_quat_w

        # Convert quaternion to euler to check orientation
        roll = torch.atan2(
            2 * (base_quat[:, 0] * base_quat[:, 1] + base_quat[:, 2] * base_quat[:, 3]),
            1 - 2 * (base_quat[:, 1] ** 2 + base_quat[:, 2] ** 2),
        )
        pitch = torch.asin(
            2 * (base_quat[:, 0] * base_quat[:, 2] - base_quat[:, 3] * base_quat[:, 1])
        )

        fallen = (
            (base_height < self.cfg.termination_height)
            | (torch.abs(roll) > 1.0)
            | (torch.abs(pitch) > 1.0)  # Match Isaac Gym termination height
        )

        # Check if robot is unstable (e.g., too much velocity)
        unstable = (
            torch.norm(self.robot.data.root_lin_vel_w, dim=1)
            > self.cfg.termination_velocity
        )

        died = fallen | unstable
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # Reset robot state
        # Randomize initial position slightly
        root_pos = self.robot.data.default_root_state[env_ids, :3]
        root_pos[:, 0] += (
            torch.rand(len(env_ids), device=self.device) - 0.5
        ) * 1.0  # [-0.5, 0.5]
        root_pos[:, 1] += (
            torch.rand(len(env_ids), device=self.device) - 0.5
        ) * 1.0  # [-0.5, 0.5]
        root_pos[:, 2] += (
            torch.rand(len(env_ids), device=self.device) - 0.5
        ) * 0.2  # [-0.1, 0.1]

        # Randomize initial orientation slightly
        root_quat = self.robot.data.default_root_state[env_ids, 3:7]
        root_quat[:, 3] += (
            torch.rand(len(env_ids), device=self.device) - 0.5
        ) * 0.2  # [-0.1, 0.1]
        root_quat = torch.nn.functional.normalize(root_quat, dim=1)

        # Randomize joint positions slightly around default
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos += (
            torch.rand(joint_pos.shape, device=self.device) - 0.5
        ) * 0.2  # [-0.1, 0.1]

        # Set velocities to zero
        root_vel = torch.zeros((len(env_ids), 6), device=self.device)
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])

        # Apply the reset - Isaac Lab API expects concatenated pose (pos + quat) and velocity
        root_pose = torch.cat([root_pos, root_quat], dim=-1)  # [N, 7] (pos + quat)
        self.robot.write_root_pose_to_sim(root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset tracking variables
        if self.last_dof_vel is None:
            # Initialize with correct number of DOFs
            num_dofs = self.robot.data.joint_vel.shape[-1]
            self.last_dof_vel = torch.zeros(
                (self.num_envs, num_dofs), device=self.device
            )
        if not hasattr(self, "last_actions"):
            self.last_actions = torch.zeros(
                (self.num_envs, self.cfg.action_space), device=self.device
            )
        if not hasattr(self, "last_root_vel"):
            self.last_root_vel = torch.zeros((self.num_envs, 6), device=self.device)

        self.last_dof_vel[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_root_vel[env_ids] = 0.0

        # Reset velocity filters
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0

        # Reset gait tracking
        self.gait_process[env_ids] = 0.0
        # Randomize gait frequency (1.0 to 2.0 Hz like Isaac Gym)
        self.gait_frequency[env_ids] = 1.0 + torch.rand(
            len(env_ids), device=self.device
        )

        # Reset episode tracking
        for key in self.episode_sums.keys():
            self.episode_sums[key][env_ids] = 0.0

        # Generate new commands for reset environments
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        """Generate new random commands for the specified environments."""
        # Random linear velocity command
        self.commands[env_ids, 0] = (
            (torch.rand(len(env_ids), device=self.device) - 0.5)
            * 2
            * self.cfg.max_absolute_command_vel_x
        )
        self.commands[env_ids, 1] = (
            (torch.rand(len(env_ids), device=self.device) - 0.5)
            * 2
            * self.cfg.max_absolute_command_vel_y
        )

        # Random angular velocity command
        self.commands[env_ids, 2] = (
            (torch.rand(len(env_ids), device=self.device) - 0.5)
            * 2
            * self.cfg.max_absolute_command_angular_speed
        )
