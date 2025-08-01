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
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import subtract_frame_transforms

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

        # Command tracking - max_episode_length is automatically set by Isaac Lab

        # Rewards tracking
        self.episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }

        # Command generators
        self._init_command_terms()

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

        # Contact body indices
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
                return  # Exit early if there are issues

        self.contact_body_indices = torch.tensor(
            self.contact_body_indices, dtype=torch.long, device=self.device
        )

        print(
            f"Successfully initialized {len(self.leg_dof_indices)} joint indices and {len(self.contact_body_indices)} contact body indices"
        )

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by the inverse of quaternion q."""
        q_w = q[:, 0]
        q_vec = q[:, 1:]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
        return a - b + c

    def _init_command_terms(self):
        """Initialize command tracking variables."""
        self.commands = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.command_sums = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

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
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply the processed actions to the robot."""
        # Apply actions as joint position targets to leg joints only
        if self.leg_dof_indices is not None and len(self.leg_dof_indices) > 0:
            self.robot.set_joint_position_target(
                self.actions, joint_ids=self.leg_dof_indices
            )

    def _get_observations(self) -> dict:
        """Compute observations for the environment."""
        # Base observations
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_quat = self.robot.data.root_quat_w

        # Joint states - use all joints if indices not ready
        if self.leg_dof_indices is not None:
            joint_pos = self.robot.data.joint_pos[:, self.leg_dof_indices]
            joint_vel = self.robot.data.joint_vel[:, self.leg_dof_indices]
        else:
            # Use first 12 joints as default (assumes leg joints are first)
            joint_pos = self.robot.data.joint_pos[:, :12]
            joint_vel = self.robot.data.joint_vel[:, :12]

        # Commands - all 3 commands (x,y linear velocity + angular velocity)
        commands = self.commands

        # Concatenate observations
        obs = torch.cat(
            [
                base_lin_vel,  # 3
                base_ang_vel,  # 3
                base_quat,  # 4
                joint_pos,  # 12
                joint_vel,  # 12
                commands,  # 3
                self.actions,  # 12 (previous actions)
            ],
            dim=-1,
        )
        # Total: 3 + 3 + 4 + 12 + 12 + 3 + 12 = 49 ✓

        # Add noise if specified
        if hasattr(self.cfg, "noise_scale_vec"):
            obs += torch.randn_like(obs) * self.cfg.noise_scale_vec.to(obs.device)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward for the current step."""
        # Linear velocity tracking reward
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        lin_vel_reward = (
            torch.exp(-lin_vel_error / 0.25) * self.cfg.lin_vel_reward_scale
        )

        # Angular velocity tracking reward - use the angular command (third element)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        )
        ang_vel_reward = (
            torch.exp(-ang_vel_error / 0.25) * self.cfg.ang_vel_reward_scale
        )

        # Torque penalty
        torque_penalty = (
            torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
            * self.cfg.torque_reward_scale
        )

        # Joint acceleration penalty
        if self.leg_dof_indices is not None:
            joint_acc = (
                self.robot.data.joint_vel[:, self.leg_dof_indices] - self.last_joint_vel
            ) / self.step_dt
        else:
            # Use first 12 joints as fallback
            joint_acc = (
                self.robot.data.joint_vel[:, :12] - self.last_joint_vel
            ) / self.step_dt
        joint_acc_penalty = (
            torch.sum(torch.square(joint_acc), dim=1)
            * self.cfg.joint_accel_reward_scale
        )

        # Action rate penalty
        action_rate_penalty = (
            torch.sum(torch.square(self.actions - self.last_actions), dim=1)
            * self.cfg.action_rate_reward_scale
        )

        # Orientation penalty (keep robot upright)
        base_quat = self.robot.data.root_quat_w
        up_proj = 2 * (
            base_quat[:, 1] * base_quat[:, 3] + base_quat[:, 0] * base_quat[:, 2]
        )
        orientation_penalty = (
            torch.square(up_proj) * self.cfg.flat_orientation_reward_scale
        )

        # Sum all rewards
        total_reward = (
            lin_vel_reward
            + ang_vel_reward
            + torque_penalty
            + joint_acc_penalty
            + action_rate_penalty
            + orientation_penalty
        )

        # Store for tracking
        self.episode_sums["track_lin_vel_xy_exp"] += lin_vel_reward
        self.episode_sums["track_ang_vel_z_exp"] += ang_vel_reward
        self.episode_sums["dof_torques_l2"] += -torque_penalty
        self.episode_sums["dof_acc_l2"] += -joint_acc_penalty
        self.episode_sums["action_rate_l2"] += -action_rate_penalty
        self.episode_sums["flat_orientation_l2"] += -orientation_penalty

        # Update tracking variables
        if self.leg_dof_indices is not None:
            self.last_joint_vel = self.robot.data.joint_vel[
                :, self.leg_dof_indices
            ].clone()
        else:
            self.last_joint_vel = self.robot.data.joint_vel[:, :12].clone()
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
            (base_height < 0.3) | (torch.abs(roll) > 1.0) | (torch.abs(pitch) > 1.0)
        )

        died = fallen
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
        if not hasattr(self, "last_joint_vel") or self.leg_dof_indices is None:
            # Initialize with safe defaults if joint indices not ready yet
            if self.leg_dof_indices is not None:
                self.last_joint_vel = torch.zeros(
                    (self.num_envs, len(self.leg_dof_indices)), device=self.device
                )
            else:
                self.last_joint_vel = torch.zeros(
                    (self.num_envs, 12), device=self.device
                )  # Default to 12 leg joints
        if not hasattr(self, "last_actions"):
            self.last_actions = torch.zeros(
                (self.num_envs, self.cfg.action_space), device=self.device
            )

        self.last_joint_vel[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

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
            * self.cfg.max_command_speed
        )
        self.commands[env_ids, 1] = (
            (torch.rand(len(env_ids), device=self.device) - 0.5)
            * 2
            * self.cfg.max_command_speed
        )

        # Random angular velocity command
        self.commands[env_ids, 2] = (
            (torch.rand(len(env_ids), device=self.device) - 0.5)
            * 2
            * self.cfg.max_command_angular_speed
        )
