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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass


@configclass
class T1EnvCfg(DirectRLEnvCfg):
    """Configuration for the T1 humanoid locomotion environment."""

    # env
    episode_length_s = 30.0  # Match Isaac Gym's episode length
    decimation = 10  # Match Isaac Gym's control decimation * substeps
    num_actions = 12
    num_observations = 47  # Match Isaac Gym: 3+3+3+2+12+12+12 = 47
    num_states = 0  # No asymmetric observations for now

    # Define the action and observation spaces properly
    action_space = 12
    observation_space = 47  # Match Isaac Gym: 3+3+3+2+12+12+12 = 47
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,  # 500 Hz to match Isaac Gym's 0.002 timestep / 1 substep
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=4,  # Match Isaac Gym's position iteration count
            max_velocity_iteration_count=1,  # Match Isaac Gym's velocity iteration count
            bounce_threshold_velocity=2.0,  # Match Isaac Gym's bounce threshold
            gpu_max_rigid_contact_count=8388608,  # Match Isaac Gym's GPU contact count
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        env_spacing=4.0,
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
                solver_velocity_iteration_count=1,
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
                stiffness={
                    ".*_Hip_.*": 200.0,
                    ".*_Knee_.*": 200.0,
                    ".*_Ankle_.*": 50.0,
                },
                damping={
                    ".*_Hip_.*": 5.0,
                    ".*_Knee_.*": 5.0,
                    ".*_Ankle_.*": 1.0,
                },
            ),
        },
        soft_joint_pos_limit_factor=0.9,
    )

    collision_sensors = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/(?:Trunk|H1|H2|AL|AR|Waist|Hip|Shank|Ankle).*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        # filter_prim_paths_expr=[
        #     "/World/envs/env_.*/Robot/(?:Trunk|H1|H2|AL|AR|Waist|Hip|Shank|Ankle).*",
        # ],
    )

    foot_contact_sensors = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/(?:left|right)_foot.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        # filter_prim_paths_expr=[
        #     "/World/ground",
        # ],
    )
    # note: turns out filtering with the ground isn't supported because it's not a rigid body?

    # bodies list necessary to calculate foot slippage (contact doesn't have state)
    foot_contact_bodies = [
        "left_foot_link",
        "right_foot_link",
    ]

    # reward scales (matching Isaac Gym values exactly)
    survival_reward_scale = 0.25
    tracking_lin_vel_x_reward_scale = 1.0
    tracking_lin_vel_y_reward_scale = 1.0
    tracking_ang_vel_reward_scale = 0.5
    base_height_reward_scale = -20.0
    orientation_reward_scale = -5.0
    torques_reward_scale = -2.0e-4
    torque_tiredness_reward_scale = -1.0e-2
    power_reward_scale = -2.0e-3
    lin_vel_z_reward_scale = -2.0
    ang_vel_xy_reward_scale = -0.2
    dof_vel_reward_scale = -1.0e-4
    dof_acc_reward_scale = -1.0e-7
    root_acc_reward_scale = -1.0e-4
    action_rate_reward_scale = -1.0
    dof_pos_limits_reward_scale = -1.0
    dof_vel_limits_reward_scale = 0.0  # Disabled in Isaac Gym
    torque_limits_reward_scale = 0.0  # Disabled in Isaac Gym
    collision_reward_scale = -1.0
    feet_slip_reward_scale = -0.1
    feet_vel_z_reward_scale = 0.0  # Disabled in Isaac Gym
    feet_yaw_diff_reward_scale = -1.0
    feet_yaw_mean_reward_scale = -1.0
    feet_roll_reward_scale = -0.1
    feet_distance_reward_scale = -1.0
    feet_swing_reward_scale = 3.0
    only_positive_rewards = True  # Match Isaac Gym's positive rewards behavior

    # normalization
    gravity_normalization = 1.0
    lin_vel_normalization = 1.0
    ang_vel_normalization = 1.0
    dof_pos_normalization = 1.0
    dof_vel_normalization = 0.1
    filter_weight = 0.1
    clip_actions = 1.0  # control parameters (matching Isaac Gym)
    action_scale = 1.0

    # randomization
    # push_force_normalization = 0.1 # wrong, shoudl be randomization
    # push_torque_normalization = 0.5 # wrong, should be randomization

    # tracking
    tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
    base_height_target = 0.68
    soft_dof_pos_limit = 1.0  # percentage of urdf limits
    soft_dof_vel_limit = 1.0
    soft_torque_limit = 1.0

    # gait parameters
    swing_period = 0.2
    feet_distance_ref = 0.2

    # command ranges
    max_absolute_command_vel_x = 1.0  # Match Isaac Gym's command speed range
    max_absolute_command_vel_y = 1.0  # Match Isaac Gym's command speed range
    max_absolute_command_angular_speed = 1.0  # Match Isaac Gym's command angular speed

    # contact body names for reward calculation (use actual USD link names)
    contact_bodies = ["left_foot_link", "right_foot_link"]

    # termination conditions
    termination_height = 0.45
    termination_velocity = 50  # Match Isaac Gym's termination velocity
