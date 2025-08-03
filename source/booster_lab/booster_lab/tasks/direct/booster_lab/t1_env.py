"""
T1 humanoid locomotion environment for Isaac Lab.

This environment is based on the original Isaac Gym T1 implementation and adapted for Isaac Lab.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING
from enum import Enum

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import euler_xyz_from_quat
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
        # TODO these may be redundant due to isaac property default_joint_pos
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
                "collisions",
                "lin_vel_z",
                "ang_vel_xy",
                "orientation",
                "torques",
                "dof_vel",
                "dof_acc",
                "root_acc",
                "action_rate",
                "dof_pos_limits",
                "power",
                "feet_slip",
                "feet_roll",
                "feet_yaw_diff",
                "feet_yaw_mean",
                "feet_distance",
                "feet_swing",
            ]
        }

    # TODO, there's probably a better way to do everything in this function
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
        self.robot = Articulation(self.cfg.robot)
        self.terrain = TerrainImporter(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.collision_sensors = ContactSensor(self.cfg.collision_sensors)
        self.scene.sensors["collision_sensors"] = self.collision_sensors
        self.foot_contact_sensors = ContactSensor(self.cfg.foot_contact_sensors)
        self.scene.sensors["foot_contact_sensors"] = self.foot_contact_sensors

    def reset(self, seed=None, options=None):
        """Override reset to initialize joint indices after first reset."""
        # TODO this might be moved to setup scene or deleted
        if self.contact_body_indices is None:
            self._get_joint_indices()

        # Call parent reset
        result = super().reset(seed=seed, options=options)

        return result

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
        self.feet_pos = self.robot.data.body_link_pos_w[:, self.contact_body_indices]
        self.feet_quat = self.robot.data.body_link_quat_w[:, self.contact_body_indices]
        roll, _, yaw = euler_xyz_from_quat(self.feet_quat.reshape(-1, 4))
        self.feet_roll = (
            roll.reshape(self.num_envs, len(self.contact_body_indices)) + torch.pi
        ) % (2 * torch.pi) - torch.pi
        self.feet_yaw = (
            yaw.reshape(self.num_envs, len(self.contact_body_indices)) + torch.pi
        ) % (2 * torch.pi) - torch.pi

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

        SURVIVAL = 0
        TRACKING_LIN_VEL_X = 1
        TRACKING_LIN_VEL_Y = 2
        TRACKING_ANG_VEL = 3
        BASE_HEIGHT = 4
        COLLISIONS = 5
        LIN_VEL_Z = 6
        ANG_VEL_XY = 7
        ORIENTATION = 8
        TORQUES = 9
        DOF_VEL = 10
        DOF_ACC = 11
        ROOT_ACC = 12
        ACTION_RATE = 13
        DOF_POS_LIMITS = 14
        POWER = 15
        FEET_SLIP = 16
        FEET_ROLL = 17
        FEET_YAW_DIFF = 18
        FEET_YAW_MEAN = 19
        FEET_DISTANCE = 20
        FEET_SWING = 21
        NUM_REWARD_TERMS = 22

        reward_terms = torch.zeros(self.num_envs, NUM_REWARD_TERMS, device=self.device)

        reward_terms[:, SURVIVAL] = self.cfg.survival_reward_scale

        lin_vel_x_error = torch.square(
            self.commands[:, 0] - self.filtered_lin_vel[:, 0]
        )
        reward_terms[:, TRACKING_LIN_VEL_X] = (
            torch.exp(-lin_vel_x_error / self.cfg.tracking_sigma)
            * self.cfg.tracking_lin_vel_x_reward_scale
        )

        lin_vel_y_error = torch.square(
            self.commands[:, 1] - self.filtered_lin_vel[:, 1]
        )
        reward_terms[:, TRACKING_LIN_VEL_Y] = (
            torch.exp(-lin_vel_y_error / self.cfg.tracking_sigma)
            * self.cfg.tracking_lin_vel_y_reward_scale
        )

        ang_vel_error = torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2])
        reward_terms[:, TRACKING_ANG_VEL] = (
            torch.exp(-ang_vel_error / self.cfg.tracking_sigma)
            * self.cfg.tracking_ang_vel_reward_scale
        )

        base_height = self.robot.data.root_pos_w[:, 2]
        height_error = torch.square(base_height - self.cfg.base_height_target)
        reward_terms[:, BASE_HEIGHT] = height_error * self.cfg.base_height_reward_scale

        collision_body_violations = (
            torch.norm(self.collision_sensors.data.net_forces_w, dim=-1) > 1.0
        )
        num_collision_violations = collision_body_violations.sum(dim=-1)
        reward_terms[:, COLLISIONS] = (
            num_collision_violations * self.cfg.collision_reward_scale
        )

        reward_terms[:, LIN_VEL_Z] = (
            torch.square(self.filtered_lin_vel[:, 2]) * self.cfg.lin_vel_z_reward_scale
        )

        reward_terms[:, ANG_VEL_XY] = (
            torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=-1)
            * self.cfg.ang_vel_xy_reward_scale
        )

        reward_terms[:, ORIENTATION] = (
            torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
            * self.cfg.orientation_reward_scale
        )

        reward_terms[:, TORQUES] = (
            torch.sum(torch.square(self.robot.data.applied_torque), dim=-1)
            * self.cfg.torques_reward_scale
        )

        dof_vel = self.robot.data.joint_vel
        reward_terms[:, DOF_VEL] = (
            torch.sum(torch.square(dof_vel), dim=-1) * self.cfg.dof_vel_reward_scale
        )

        joint_acc = (self.last_dof_vel - dof_vel) / self.step_dt
        reward_terms[:, DOF_ACC] = (
            torch.sum(torch.square(joint_acc), dim=-1) * self.cfg.dof_acc_reward_scale
        )

        root_lin_acc = (
            self.last_filtered_lin_vel - self.filtered_lin_vel
        ) / self.step_dt
        root_ang_acc = (
            self.last_filtered_ang_vel - self.filtered_ang_vel
        ) / self.step_dt
        reward_terms[:, ROOT_ACC] = (
            torch.sum(torch.square(root_lin_acc), dim=-1)
            + torch.sum(torch.square(root_ang_acc), dim=-1)
        ) * self.cfg.root_acc_reward_scale

        reward_terms[:, ACTION_RATE] = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=-1)
            * self.cfg.action_rate_reward_scale
        )

        # DOF position limits penalty - fixed shape handling
        joint_pos_limits_lower = self.robot.data.joint_pos_limits[0, :, 0]
        joint_pos_limits_upper = self.robot.data.joint_pos_limits[0, :, 1]
        lower = joint_pos_limits_lower + 0.5 * (1 - self.cfg.soft_dof_pos_limit) * (
            joint_pos_limits_upper - joint_pos_limits_lower
        )
        upper = joint_pos_limits_upper - 0.5 * (1 - self.cfg.soft_dof_pos_limit) * (
            joint_pos_limits_upper - joint_pos_limits_lower
        )
        reward_terms[:, DOF_POS_LIMITS] = (
            torch.sum(
                (
                    (self.robot.data.joint_pos < lower)
                    | (self.robot.data.joint_pos > upper)
                ).float(),
                dim=-1,
            )
            * self.cfg.dof_pos_limits_reward_scale
        )

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

        reward_terms[:, POWER] = torch.sum(
            (self.robot.data.applied_torque * self.robot.data.joint_vel).clip(min=0.0),
            dim=-1,
        )

        feet_movement = torch.square(
            torch.sum(((self.last_feet_pos - self.feet_pos) / self.step_dt), dim=-1)
        )
        feet_ground_contact = torch.sum(self.foot_contact_sensors.data.net_forces_w, dim=-1) > 0.0
        feet_slip = torch.sum(feet_movement * feet_ground_contact, dim=-1)
        reward_terms[:, FEET_SLIP] = (
            feet_slip
            * self.cfg.feet_slip_reward_scale
            * (self.episode_length_buf > 1).float()
        )

        # missed feet vel z
        # (disabled)

        feet_roll = torch.sum(torch.square(self.feet_roll), dim=-1)
        reward_terms[:, FEET_ROLL] = feet_roll * self.cfg.feet_roll_reward_scale

        feet_yaw_diff = torch.square(
            (self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi)
            - torch.pi
        )
        reward_terms[:, FEET_YAW_DIFF] = (
            feet_yaw_diff * self.cfg.feet_yaw_diff_reward_scale
        )

        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (
            torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi
        )
        reward_terms[:, FEET_YAW_MEAN] = (
            torch.square(
                (
                    euler_xyz_from_quat(self.robot.data.root_quat_w)[2]
                    - feet_yaw_mean
                    + torch.pi
                )
                % (2 * torch.pi)
                - torch.pi
            )
            * self.cfg.feet_yaw_mean_reward_scale
        )

        # TODO probably squared reward
        _, _, base_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        reward_terms[:, FEET_DISTANCE] = (
            torch.clip(
                self.cfg.feet_distance_reward_scale - feet_distance, min=0.0, max=1.0
            )
            * self.cfg.feet_distance_reward_scale
        )

        left_swing = (
            torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg.swing_period
        ) & (self.gait_frequency > 1e-8)
        right_swing = (
            torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg.swing_period
        ) & (self.gait_frequency > 1e-8)
        reward_terms[:, FEET_SWING] = (
            left_swing & ~feet_ground_contact[:, 0]
        ).float() + (
            right_swing & ~feet_ground_contact[:, 1]
        ).float() * self.cfg.feet_swing_reward_scale

        # Apply optional positive-only clipping to all reward terms if configured
        if self.cfg.only_positive_rewards:
            reward_terms = torch.clip(reward_terms, min=0.0)

        # Store episode sums for tracking (using individual terms from the tensor)
        self.episode_sums["survival"] += reward_terms[:, SURVIVAL]
        self.episode_sums["tracking_lin_vel_x"] += reward_terms[:, TRACKING_LIN_VEL_X]
        self.episode_sums["tracking_lin_vel_y"] += reward_terms[:, TRACKING_LIN_VEL_Y]
        self.episode_sums["tracking_ang_vel"] += reward_terms[:, TRACKING_ANG_VEL]
        self.episode_sums["base_height"] += reward_terms[:, BASE_HEIGHT]
        self.episode_sums["collisions"] += reward_terms[:, COLLISIONS]
        self.episode_sums["lin_vel_z"] += reward_terms[:, LIN_VEL_Z]
        self.episode_sums["ang_vel_xy"] += reward_terms[:, ANG_VEL_XY]
        self.episode_sums["orientation"] += reward_terms[:, ORIENTATION]
        self.episode_sums["torques"] += reward_terms[:, TORQUES]
        self.episode_sums["dof_vel"] += reward_terms[:, DOF_VEL]
        self.episode_sums["dof_acc"] += reward_terms[:, DOF_ACC]
        self.episode_sums["root_acc"] += reward_terms[:, ROOT_ACC]
        self.episode_sums["action_rate"] += reward_terms[:, ACTION_RATE]
        self.episode_sums["dof_pos_limits"] += reward_terms[:, DOF_POS_LIMITS]
        self.episode_sums["power"] += reward_terms[:, POWER]
        self.episode_sums["feet_slip"] += reward_terms[:, FEET_SLIP]
        self.episode_sums["feet_roll"] += reward_terms[:, FEET_ROLL]
        self.episode_sums["feet_yaw_diff"] += reward_terms[:, FEET_YAW_DIFF]
        self.episode_sums["feet_yaw_mean"] += reward_terms[:, FEET_YAW_MEAN]
        self.episode_sums["feet_distance"] += reward_terms[:, FEET_DISTANCE]
        self.episode_sums["feet_swing"] += reward_terms[:, FEET_SWING]

        # Sum all reward terms efficiently: shape (num_envs,)
        total_reward = torch.sum(reward_terms, dim=1)

        # Update previous states
        # TODO this should be moved
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
