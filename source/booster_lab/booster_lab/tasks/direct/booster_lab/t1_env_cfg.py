"""Configuration for the T1 humanoid locomotion environment."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class T1EnvCfg(DirectRLEnvCfg):
    """Configuration for the T1 humanoid locomotion environment."""

    # env
    episode_length_s = 5.0
    decimation = 2
    num_actions = 12
    num_observations = 49  # Updated to match actual observation size
    num_states = 0  # No asymmetric observations for now

    # Define the action and observation spaces properly
    action_space = 12
    observation_space = 49  # Updated to match actual observation size
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 120 Hz (decimation=2 gives effective 60 Hz)
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # physx=sim_utils.PhysxCfg(
        #     solver_type=1,  # TGS
        # ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.0, replicate_physics=True
    )

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        env_spacing=1.0,  # Add environment spacing for grid-like origins
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"C:/Users/reill/booster_gym/booster_lab/source/booster_lab/booster_lab/assets/t1/t1.usd",  # absolute filepath
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.72),  # Use same initial height as original config
            joint_pos={
                # T1 initial joint positions using actual URDF joint names
                "Left_Hip_Pitch": -0.2,
                "Left_Hip_Roll": 0.0,
                "Left_Hip_Yaw": 0.0,
                "Left_Knee_Pitch": 0.4,
                "Left_Ankle_Pitch": -0.25,
                "Left_Ankle_Roll": 0.0,
                "Right_Hip_Pitch": -0.2,
                "Right_Hip_Roll": 0.0,
                "Right_Hip_Yaw": 0.0,
                "Right_Knee_Pitch": 0.4,
                "Right_Ankle_Pitch": -0.25,
                "Right_Ankle_Roll": 0.0,
            },
            joint_vel={
                # Initialize all joint velocities to zero
                "Left_Hip_Pitch": 0.0,
                "Left_Hip_Roll": 0.0,
                "Left_Hip_Yaw": 0.0,
                "Left_Knee_Pitch": 0.0,
                "Left_Ankle_Pitch": 0.0,
                "Left_Ankle_Roll": 0.0,
                "Right_Hip_Pitch": 0.0,
                "Right_Hip_Roll": 0.0,
                "Right_Hip_Yaw": 0.0,
                "Right_Knee_Pitch": 0.0,
                "Right_Ankle_Pitch": 0.0,
                "Right_Ankle_Roll": 0.0,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_Hip_.*",  # Matches all hip joints
                    ".*_Knee_.*",  # Matches all knee joints
                    ".*_Ankle_.*",  # Matches all ankle joints
                ],
                effort_limit=400.0,
                velocity_limit=30.0,
                stiffness=80.0,
                damping=2.0,
            ),
        },
        soft_joint_pos_limit_factor=0.9,
    )

    # reward scales
    lin_vel_reward_scale = 1.0
    ang_vel_reward_scale = 0.5
    torque_reward_scale = -0.0002
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 1.0
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -1.0

    # command ranges
    max_command_speed = 1.5
    max_command_angular_speed = 1.0

    # contact body names for reward calculation (use actual URDF link names)
    contact_bodies = ["left_foot_link", "right_foot_link"]

    # termination conditions
    max_cart_pos = 3.0
