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

if TYPE_CHECKING:
    from .t1_env_cfg import T1EnvCfg


class T1Env(DirectRLEnv):
    """Environment for T1 humanoid locomotion."""

    cfg: T1EnvCfg

    def __init__(self, cfg: T1EnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the T1 environment."""
        super().__init__(cfg, render_mode, **kwargs)

        # Joint and body indices will be set after scene is created
        self.leg_dof_indices = None
        self.contact_body_indices = None
        self.penalized_contact_indices = None

        # Velocity filtering (matching Isaac Gym)
        self.filtered_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.filtered_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

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

        # Previous states for acceleration calculation
        self.last_dof_vel = torch.zeros(self.num_envs, 12, device=self.device)
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

        # Based on the actual URDF joint names
        self.leg_dof_names = [
            "Left_Hip_Pitch",
            "Left_Hip_Roll",
            "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Hip_Pitch",
            "Right_Hip_Roll",
            "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        ]

        self.leg_dof_indices = []
        for name in self.leg_dof_names:
            try:
                # Use find_joints method which returns (indices, names)
                idx, _ = self.robot.find_joints(name)
                if len(idx) > 0:
                    self.leg_dof_indices.append(idx[0])
                else:
                    print(f"Warning: Joint {name} not found in robot")
            except Exception as e:
                print(f"Warning: Error finding joint {name}: {e}")
                return  # Exit early if there are issues

        self.leg_dof_indices = torch.tensor(
            self.leg_dof_indices, dtype=torch.long, device=self.device
        )

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

        # Contact body indices (penalized contacts)
        self.penalized_contact_names = []
        penalized_names = [
            "Trunk",
            "H1",
            "H2",
            "AL",
            "AR",
            "Waist",
            "Hip",
            "Shank",
            "Ankle",
        ]
        body_names = self.robot.body_names
        for name in penalized_names:
            self.penalized_contact_names.extend([s for s in body_names if name in s])

        self.penalized_contact_indices = []
        for name in self.penalized_contact_names:
            try:
                idx, _ = self.robot.find_bodies(name)
                if len(idx) > 0:
                    self.penalized_contact_indices.append(idx[0])
            except Exception as e:
                print(f"Warning: Error finding penalized contact body {name}: {e}")

        if len(self.penalized_contact_indices) > 0:
            self.penalized_contact_indices = torch.tensor(
                self.penalized_contact_indices, dtype=torch.long, device=self.device
            )
        else:
            self.penalized_contact_indices = None

        print(
            f"Successfully initialized {len(self.leg_dof_indices)} joint indices, {len(self.contact_body_indices)} contact body indices, and {len(self.penalized_contact_indices) if self.penalized_contact_indices is not None else 0} penalized contact indices"
        )

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

    def reset(self, seed=None, options=None):
        """Override reset to initialize joint indices after first reset."""
        # Initialize joint indices before any reset logic if not already done
        if self.leg_dof_indices is None:
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
        if self.leg_dof_indices is not None and len(self.leg_dof_indices) > 0:
            dof_targets = self.default_dof_pos + self.cfg.action_scale * self.actions
            self.robot.set_joint_position_target(
                dof_targets, joint_ids=self.leg_dof_indices
            )

    def _get_observations(self) -> dict:
        """Compute observations for the environment - match Isaac Gym exactly."""
        # Update filtered velocities (exponential moving average)
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b

        self.filtered_lin_vel[:] = (
            base_lin_vel * self.cfg.filter_weight
            + self.filtered_lin_vel * (1.0 - self.cfg.filter_weight)
        )
        self.filtered_ang_vel[:] = (
            base_ang_vel * self.cfg.filter_weight
            + self.filtered_ang_vel * (1.0 - self.cfg.filter_weight)
        )

        # Projected gravity (gravity in robot frame)
        base_quat = self.robot.data.root_quat_w
        self.projected_gravity[:] = self._quat_rotate_inverse(
            base_quat, self.gravity_vec
        )

        # Joint states
        if self.leg_dof_indices is not None:
            joint_pos = self.robot.data.joint_pos[:, self.leg_dof_indices]
            joint_vel = self.robot.data.joint_vel[:, self.leg_dof_indices]
            # Use pre-computed default positions
            default_joint_pos = self.default_dof_pos
        else:
            joint_pos = self.robot.data.joint_pos[:, :12]
            joint_vel = self.robot.data.joint_vel[:, :12]
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

        # Survival reward
        survival_rew = self.cfg.survival_reward_scale
        total_reward += survival_rew

        # Linear velocity tracking rewards (X and Y separately like Isaac Gym)
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

        # Angular velocity tracking reward
        ang_vel_error = torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2])
        ang_vel_rew = (
            torch.exp(-ang_vel_error / self.cfg.tracking_sigma)
            * self.cfg.tracking_ang_vel_reward_scale
        )
        total_reward += ang_vel_rew

        # Base height penalty
        base_height = self.robot.data.root_pos_w[:, 2]
        height_error = torch.square(base_height - self.cfg.base_height_target)
        height_penalty = height_error * self.cfg.base_height_reward_scale
        total_reward += height_penalty

        # Orientation penalty (keep robot upright)
        orientation_penalty = (
            torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
            * self.cfg.orientation_reward_scale
        )
        total_reward += orientation_penalty

        # Torque penalty
        torque_penalty = (
            torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
            * self.cfg.torques_reward_scale
        )
        total_reward += torque_penalty

        # DOF velocity penalty
        if self.leg_dof_indices is not None:
            joint_vel = self.robot.data.joint_vel[:, self.leg_dof_indices]
        else:
            joint_vel = self.robot.data.joint_vel[:, :12]
        dof_vel_penalty = (
            torch.sum(torch.square(joint_vel), dim=1) * self.cfg.dof_vel_reward_scale
        )
        total_reward += dof_vel_penalty

        # DOF acceleration penalty
        joint_acc = (joint_vel - self.last_dof_vel) / self.step_dt
        dof_acc_penalty = (
            torch.sum(torch.square(joint_acc), dim=1) * self.cfg.dof_acc_reward_scale
        )
        total_reward += dof_acc_penalty

        # Action rate penalty
        action_rate_penalty = torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )
        total_reward += action_rate_penalty

        # Linear velocity Z penalty (discourage jumping/falling)
        lin_vel_z_penalty = (
            torch.square(self.filtered_lin_vel[:, 2]) * self.cfg.lin_vel_z_reward_scale
        )
        total_reward += lin_vel_z_penalty

        # Angular velocity XY penalty (discourage roll/pitch)
        ang_vel_xy_penalty = (
            torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
            * self.cfg.ang_vel_xy_reward_scale
        )
        total_reward += ang_vel_xy_penalty

        # Collision penalty (undesired contacts) - simplified for now
        # TODO: Implement proper contact force detection in Isaac Lab
        collision_penalty = torch.zeros(self.num_envs, device=self.device)
        total_reward += collision_penalty

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
        self.last_dof_vel = joint_vel.clone()
        self.last_root_vel = torch.cat(
            [self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w], dim=1
        ).clone()
        self.last_actions = self.actions.clone()

        return total_reward

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
        if not hasattr(self, "last_dof_vel") or self.leg_dof_indices is None:
            # Initialize with safe defaults if joint indices not ready yet
            if self.leg_dof_indices is not None:
                self.last_dof_vel = torch.zeros(
                    (self.num_envs, len(self.leg_dof_indices)), device=self.device
                )
            else:
                self.last_dof_vel = torch.zeros(
                    (self.num_envs, 12), device=self.device
                )  # Default to 12 leg joints
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
