# Copyright (c) 2026
# Dual-arm reach with obstacle avoidance for Isaac Gym.
# Stage 1: joint position residual (dualarm-python-test joint env)
# Stage 2: end-effector delta + OSC (dualarm-python-test EE env)

import os

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import (
    quat_conjugate,
    quat_mul,
    quat_to_exp_map,
    tensor_clamp,
    to_torch,
)


@torch.jit.script
def axisangle2quat(vec, eps: float = 1e-6):
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)
    angle = torch.norm(vec, dim=-1, keepdim=True)
    quat = torch.zeros(vec.shape[0], 4, device=vec.device)
    quat[:, 3] = 1.0
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0),
    ], dim=-1)
    return quat.reshape(list(input_shape) + [4])


@torch.jit.script
def orientation_error_axis_angle(current_quat, target_quat):
    # Quaternions are xyzw.
    q_err = quat_mul(target_quat, quat_conjugate(current_quat))
    return quat_to_exp_map(q_err)


@torch.jit.script
def compute_dualarm_reach_reward(
    reset_buf,
    progress_buf,
    actions,
    dist_left,
    dist_right,
    prev_dist_left,
    prev_dist_right,
    obstacle_penalty_sum,
    collision,
    task_success,
    prev_task_success,
    max_episode_length,
    progress_reward_scale,
    distance_penalty_scale,
    reach_bonus_scale,
    lagging_arm_penalty_scale,
    obstacle_penalty_scale,
    collision_penalty,
    success_bonus,
    action_penalty_scale,
    step_penalty,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor]

    progress_left = prev_dist_left - dist_left
    progress_right = prev_dist_right - dist_right

    lagging_arm_dist = torch.maximum(dist_left, dist_right)
    task_success_no_collision = task_success & (~collision)
    first_success = task_success_no_collision & (~prev_task_success)

    reward = (
        progress_reward_scale * (progress_left + progress_right)
        - distance_penalty_scale * (dist_left + dist_right)
        + reach_bonus_scale * (torch.exp(-3.0 * dist_left) + torch.exp(-3.0 * dist_right))
        - lagging_arm_penalty_scale * lagging_arm_dist
        - obstacle_penalty_scale * obstacle_penalty_sum
        - action_penalty_scale * torch.norm(actions, dim=-1)
        - step_penalty
    )

    reward = reward - collision_penalty * collision.float()
    reward = reward + success_bonus * first_success.float()

    reset_buf = torch.where(
        collision | task_success_no_collision,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reset_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        reset_buf,
    )

    return reward, reset_buf, task_success_no_collision


class DualArmReach(VecTask):
    """Bimanual reach task with spherical obstacles."""

    N_ARMS = 2
    N_JOINTS = 15
    EE_BODY_NAMES = ("lewis_fr3_link7", "richard_fr3_link7")
    ACTOR_NAME = "dualarm"
    BASE_JOINT_INDEX = 0

    @classmethod
    def _num_observations(cls, num_joints, num_arms=2):
        # q, dq, ee_pos, ee_quat, target_pos_err, target_ori_err, obstacle_err
        return (
            2 * num_joints
            + 3 * num_arms
            + 4 * num_arms
            + 3 * num_arms
            + 3 * num_arms
            + 3 * num_arms
        )

    @classmethod
    def _num_actions(cls, control_type, num_joints):
        if control_type == "joint_pos":
            return num_joints
        if control_type == "osc":
            return 6 * cls.N_ARMS
        raise ValueError(f"Unsupported controlType: {control_type}")

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"joint_pos", "osc"}, (
            "Invalid controlType. Must be one of: {joint_pos, osc}"
        )

        self.action_scale = float(self.cfg["env"]["actionScale"])
        self.action_ori_scale = float(self.cfg["env"].get("actionOriScale", 0.05))
        self.fix_base_joint = bool(self.cfg["env"].get("fixBaseJoint", False))

        self.joint_pos_stiffness = float(self.cfg["env"].get("jointPosStiffness", 400.0))
        self.joint_pos_damping = float(self.cfg["env"].get("jointPosDamping", 40.0))

        self.cartesian_kp = float(self.cfg["env"].get("cartesianKp", 600.0))
        self.cartesian_kd = float(self.cfg["env"].get("cartesianKd", 30.0))
        self.cartesian_ori_kp = float(self.cfg["env"].get("cartesianOriKp", 1.0))
        self.cartesian_ori_kd = float(self.cfg["env"].get("cartesianOriKd", 0.1))
        self.null_space_kp = float(self.cfg["env"].get("nullSpaceKp", 50.0))
        self.null_space_kd = float(self.cfg["env"].get("nullSpaceKd", 5.0))
        self.use_null_space = bool(self.cfg["env"].get("useNullSpace", True))
        self.orientation_mode = self.cfg["env"].get("orientationMode", "track_target")

        self.goal_arrival_threshold = float(self.cfg["env"]["goalArrivalThreshold"])
        self.goal_arrival_ori_threshold = float(self.cfg["env"]["goalArrivalOriThreshold"])
        self.ee_radius = float(self.cfg["env"]["eeRadius"])
        self.safe_distance = float(self.cfg["env"]["safeDistance"])

        self.reward_settings = {
            "progress_reward_scale": float(self.cfg["env"]["progressRewardScale"]),
            "distance_penalty_scale": float(self.cfg["env"]["distancePenaltyScale"]),
            "reach_bonus_scale": float(self.cfg["env"]["reachBonusScale"]),
            "lagging_arm_penalty_scale": float(self.cfg["env"]["laggingArmPenaltyScale"]),
            "obstacle_penalty_scale": float(self.cfg["env"]["obstaclePenaltyScale"]),
            "collision_penalty": float(self.cfg["env"]["collisionPenalty"]),
            "success_bonus": float(self.cfg["env"]["successBonus"]),
            "action_penalty_scale": float(self.cfg["env"]["actionPenaltyScale"]),
            "step_penalty": float(self.cfg["env"]["stepPenalty"]),
        }

        self.target_pos_noise = to_torch(self.cfg["env"]["targetPosNoise"], device="cpu")
        self.obstacle_pos_noise = to_torch(self.cfg["env"]["obstaclePosNoise"], device="cpu")

        self.num_actions = self._num_actions(self.control_type, self.N_JOINTS)
        self.cfg["env"]["numObservations"] = self._num_observations(self.N_JOINTS, self.N_ARMS)
        self.cfg["env"]["numActions"] = self.num_actions

        self.obstacle_radius = 0.05
        self.obstacle_base_pos = to_torch(
            [[0.4, 0.15, 0.5], [0.4, -0.15, 0.5]], device="cpu", dtype=torch.float
        )
        self.target_pos_base = to_torch(
            [[0.3, 0.3, 0.6], [0.3, -0.3, 0.6]], device="cpu", dtype=torch.float
        )
        # MuJoCo wxyz -> Isaac Gym xyzw
        self.target_quat_base = to_torch(
            [
                [0.7071, 0.0, 0.0, 0.7071],
                [0.7071, 0.0, 0.0, -0.7071],
            ],
            device="cpu",
            dtype=torch.float,
        )
        self.default_dof_pos = to_torch(
            [
                0.0, 0.0, -0.7854, 0.0, -2.35621, -0.785398, 1.5708, 0.0,
                0.0, -0.7854, 0.0, -2.35621, 0.785398, 1.5708, 0.0,
            ],
            device="cpu",
            dtype=torch.float,
        )

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self._init_task_tensors()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs,
            self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs)),
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _configure_dof_properties(self, dof_props):
        drive_mode = int(
            gymapi.DOF_MODE_POS if self.control_type == "joint_pos" else gymapi.DOF_MODE_EFFORT
        )
        for i in range(self.num_robot_dofs):
            dof_props["driveMode"][i] = drive_mode
            if self.control_type == "joint_pos":
                dof_props["stiffness"][i] = self.joint_pos_stiffness
                dof_props["damping"][i] = self.joint_pos_damping
            else:
                dof_props["stiffness"][i] = 0.0
                dof_props["damping"][i] = 0.0

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_cfg = self.cfg["env"]["asset"]
        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), asset_cfg["assetRoot"])
        )
        asset_file = asset_cfg["assetFileName"]

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = int(
            gymapi.DOF_MODE_POS if self.control_type == "joint_pos" else gymapi.DOF_MODE_EFFORT
        )
        asset_options.use_mesh_materials = True

        dualarm_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(dualarm_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(dualarm_asset)

        if self.num_robot_dofs != self.N_JOINTS:
            raise RuntimeError(
                f"Expected {self.N_JOINTS} robot DOFs, got {self.num_robot_dofs}. "
                "Update N_JOINTS / default_dof_pos after verifying the URDF."
            )

        dof_props = self.gym.get_asset_dof_properties(dualarm_asset)
        self._configure_dof_properties(dof_props)

        self.dof_lower_limits = []
        self.dof_upper_limits = []
        self.dof_effort_limits = []
        for i in range(self.num_robot_dofs):
            self.dof_lower_limits.append(dof_props["lower"][i])
            self.dof_upper_limits.append(dof_props["upper"][i])
            self.dof_effort_limits.append(dof_props["effort"][i])

        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)
        self.dof_effort_limits = to_torch(self.dof_effort_limits, device=self.device)

        self.lewis_body_idx = self.gym.find_asset_rigid_body_index(
            dualarm_asset, self.EE_BODY_NAMES[0]
        )
        self.richard_body_idx = self.gym.find_asset_rigid_body_index(
            dualarm_asset, self.EE_BODY_NAMES[1]
        )

        obstacle_options = gymapi.AssetOptions()
        obstacle_options.fix_base_link = True
        obstacle_asset = self.gym.create_sphere(
            self.sim, self.obstacle_radius, obstacle_options
        )

        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.envs = []
        self.robot_handles = []
        self.obstacle_handles = []

        for env_id in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            robot_handle = self.gym.create_actor(
                env_ptr,
                dualarm_asset,
                robot_start_pose,
                self.ACTOR_NAME,
                env_id,
                1,
                0,
            )
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)
            self.gym.set_rigid_body_color(
                env_ptr, robot_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.8, 0.8)
            )

            obstacle_handles = []
            for obs_id, base_pos in enumerate(self.obstacle_base_pos):
                obstacle_pose = gymapi.Transform()
                obstacle_pose.p = gymapi.Vec3(*base_pos.tolist())
                obstacle_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                obstacle_handle = self.gym.create_actor(
                    env_ptr,
                    obstacle_asset,
                    obstacle_pose,
                    f"obstacle_{obs_id}",
                    env_id,
                    2,
                    0,
                )
                self.gym.set_rigid_body_color(
                    env_ptr,
                    obstacle_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.3, 0.3, 0.3),
                )
                obstacle_handles.append(obstacle_handle)

            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)
            self.obstacle_handles.append(obstacle_handles)

        self._init_sim_tensors()

    def _init_sim_tensors(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_robot_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_robot_dofs, 2)[..., 1]

        bodies_per_env = self.rigid_body_state.shape[0] // self.num_envs
        self.rigid_body_state_view = self.rigid_body_state.view(
            self.num_envs, bodies_per_env, 13
        )

        if self.control_type == "osc":
            jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, self.ACTOR_NAME)
            self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)
            mass_matrix_tensor = self.gym.acquire_mass_matrix_tensor(self.sim, self.ACTOR_NAME)
            self.mass_matrix = gymtorch.wrap_tensor(mass_matrix_tensor)

        self.obstacle_actor_indices = []
        for env_id in range(self.num_envs):
            env_obstacle_indices = []
            for obs_id in range(len(self.obstacle_base_pos)):
                handle = self.obstacle_handles[env_id][obs_id]
                actor_index = self.gym.get_actor_index(
                    self.envs[env_id], handle, gymapi.DOMAIN_SIM
                )
                env_obstacle_indices.append(actor_index)
            self.obstacle_actor_indices.append(env_obstacle_indices)
        self.obstacle_actor_indices = to_torch(
            self.obstacle_actor_indices, dtype=torch.int32, device=self.device
        )

        self.robot_actor_indices = to_torch(
            [
                self.gym.get_actor_index(self.envs[i], self.robot_handles[i], gymapi.DOMAIN_SIM)
                for i in range(self.num_envs)
            ],
            dtype=torch.int32,
            device=self.device,
        )

        self.pos_targets = torch.zeros(
            (self.num_envs, self.num_robot_dofs), device=self.device, dtype=torch.float
        )
        self.effort_targets = torch.zeros(
            (self.num_envs, self.num_robot_dofs), device=self.device, dtype=torch.float
        )

    def _init_task_tensors(self):
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
        )
        self.target_pos = self.target_pos_base.unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
        self.target_quat = self.target_quat_base.unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
        self.obstacle_pos = self.obstacle_base_pos.unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)

        self.ee_pos_des = torch.zeros(
            (self.num_envs, self.N_ARMS, 3), device=self.device, dtype=torch.float
        )
        self.ee_quat_des = torch.zeros(
            (self.num_envs, self.N_ARMS, 4), device=self.device, dtype=torch.float
        )
        self.ee_quat_des[..., 3] = 1.0

        self.prev_dist = torch.zeros(
            (self.num_envs, self.N_ARMS), device=self.device, dtype=torch.float
        )
        self.prev_task_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.control_type == "osc":
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)

    def _get_ee_state(self):
        lewis_state = self.rigid_body_state_view[:, self.lewis_body_idx]
        richard_state = self.rigid_body_state_view[:, self.richard_body_idx]
        ee_pos = torch.stack((lewis_state[:, :3], richard_state[:, :3]), dim=1)
        ee_quat = torch.stack((lewis_state[:, 3:7], richard_state[:, 3:7]), dim=1)
        return ee_pos, ee_quat

    def _get_obstacle_surface_distance(self, ee_pos):
        center_dist = torch.norm(
            ee_pos.unsqueeze(2) - self.obstacle_pos.unsqueeze(1),
            dim=-1,
        )
        return center_dist - self.obstacle_radius - self.ee_radius

    def _get_nearest_obstacle_error(self, ee_pos):
        errors = self.obstacle_pos.unsqueeze(1) - ee_pos.unsqueeze(2)
        nearest_ids = torch.argmin(torch.norm(errors, dim=-1), dim=-1)
        batch_ids = torch.arange(self.num_envs, device=self.device)
        arm_ids = torch.arange(self.N_ARMS, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        return errors[batch_ids.unsqueeze(1), arm_ids, nearest_ids]

    def _init_desired_ee_pose(self, env_ids, ee_pos, ee_quat):
        self.ee_pos_des[env_ids] = ee_pos
        if self.orientation_mode == "hold_initial":
            self.ee_quat_des[env_ids] = ee_quat
        else:
            self.ee_quat_des[env_ids] = self.target_quat[env_ids]

    def _apply_obstacle_states(self, env_ids):
        root_states = self.root_state_tensor.clone()
        for obs_id in range(self.obstacle_base_pos.shape[0]):
            actor_indices = self.obstacle_actor_indices[env_ids, obs_id]
            root_states[actor_indices, 0:3] = self.obstacle_pos[env_ids, obs_id]
            root_states[actor_indices, 3:7] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float
            )
            root_states[actor_indices, 7:13] = 0.0
        obstacle_actor_indices = self.obstacle_actor_indices[env_ids].reshape(-1).to(
            dtype=torch.int32
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(obstacle_actor_indices),
            len(obstacle_actor_indices),
        )

    def _mask_base_joint_action(self, actions):
        if not self.fix_base_joint:
            return actions
        masked = actions.clone()
        masked[:, self.BASE_JOINT_INDEX] = 0.0
        return masked

    def _apply_joint_position_control(self, actions):
        """Joint position residual: q_des = q + action_scale * action."""
        actions = self._mask_base_joint_action(actions)
        q_des = self.dof_pos + self.action_scale * actions
        self.pos_targets[:] = tensor_clamp(
            q_des, self.dof_lower_limits.unsqueeze(0), self.dof_upper_limits.unsqueeze(0)
        )
        if self.fix_base_joint:
            self.pos_targets[:, self.BASE_JOINT_INDEX] = self.default_dof_pos[
                self.BASE_JOINT_INDEX
            ].to(self.device)
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.pos_targets)
        )

    def _compute_impedance_torques(self):
        """Cartesian impedance control for stage 2 (OSC).

        TODO: add full null-space posture control and gravity compensation to match MuJoCo.
        """
        q = self.dof_pos
        qd = self.dof_vel
        ee_pos, ee_quat = self._get_ee_state()
        torques = torch.zeros_like(q)

        body_indices = (self.lewis_body_idx, self.richard_body_idx)
        for arm_idx, body_idx in enumerate(body_indices):
            j_arm = self.jacobian[:, body_idx, :, : self.num_robot_dofs]
            j_pos = j_arm[:, :3, :]
            j_ori = j_arm[:, 3:6, :]

            pos_error = self.ee_pos_des[:, arm_idx] - ee_pos[:, arm_idx]
            ee_lin_vel = torch.bmm(j_pos, qd.unsqueeze(-1)).squeeze(-1)
            force = self.cartesian_kp * pos_error - self.cartesian_kd * ee_lin_vel
            torques += torch.bmm(j_pos.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)

            ori_error = orientation_error_axis_angle(
                ee_quat[:, arm_idx], self.ee_quat_des[:, arm_idx]
            )
            ee_ang_vel = torch.bmm(j_ori, qd.unsqueeze(-1)).squeeze(-1)
            torque = self.cartesian_ori_kp * ori_error - self.cartesian_ori_kd * ee_ang_vel
            torques += torch.bmm(j_ori.transpose(1, 2), torque.unsqueeze(-1)).squeeze(-1)

        if self.use_null_space:
            posture_error = self.default_dof_pos.to(self.device) - q
            torques += self.null_space_kp * posture_error - self.null_space_kd * qd

        return tensor_clamp(
            torques,
            -self.dof_effort_limits.unsqueeze(0),
            self.dof_effort_limits.unsqueeze(0),
        )

    def _apply_osc_control(self, actions):
        actions_reshaped = actions.view(self.num_envs, self.N_ARMS, 6)
        pos_action = actions_reshaped[:, :, :3]
        ori_action = actions_reshaped[:, :, 3:]

        self.ee_pos_des = self.ee_pos_des + self.action_scale * pos_action
        delta_quat = axisangle2quat(self.action_ori_scale * ori_action)
        self.ee_quat_des = quat_mul(self.ee_quat_des, delta_quat)

        self.effort_targets[:] = self._compute_impedance_torques()
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self.effort_targets)
        )

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        if self.control_type == "joint_pos":
            self._apply_joint_position_control(self.actions)
        else:
            self._apply_osc_control(self.actions)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_observations(self):
        self._refresh()
        q = self.dof_pos
        qd = self.dof_vel
        ee_pos, ee_quat = self._get_ee_state()

        target_error_pos = self.target_pos - ee_pos
        target_error_ori = orientation_error_axis_angle(ee_quat, self.target_quat)
        obstacle_error = self._get_nearest_obstacle_error(ee_pos)

        self.obs_buf = torch.cat(
            (
                q,
                qd,
                ee_pos.reshape(self.num_envs, -1),
                ee_quat.reshape(self.num_envs, -1),
                target_error_pos.reshape(self.num_envs, -1),
                target_error_ori.reshape(self.num_envs, -1),
                obstacle_error.reshape(self.num_envs, -1),
            ),
            dim=-1,
        )
        return self.obs_buf

    def compute_reward(self, actions):
        ee_pos, ee_quat = self._get_ee_state()
        dist = torch.norm(self.target_pos - ee_pos, dim=-1)
        ori_dist = torch.norm(
            orientation_error_axis_angle(ee_quat, self.target_quat),
            dim=-1,
        )

        dist_left = dist[:, 0]
        dist_right = dist[:, 1]
        prev_dist_left = self.prev_dist[:, 0]
        prev_dist_right = self.prev_dist[:, 1]
        self.prev_dist = dist.detach()

        obstacle_dist = self._get_obstacle_surface_distance(ee_pos)
        obstacle_penalty = torch.clamp(self.safe_distance - obstacle_dist, min=0.0)
        obstacle_penalty_sum = obstacle_penalty.sum(dim=-1).sum(dim=-1)
        collision = (obstacle_dist < 0.0).any(dim=-1).any(dim=-1)

        success_left = (dist_left < self.goal_arrival_threshold) & (
            ori_dist[:, 0] < self.goal_arrival_ori_threshold
        )
        success_right = (dist_right < self.goal_arrival_threshold) & (
            ori_dist[:, 1] < self.goal_arrival_ori_threshold
        )
        task_success = success_left & success_right

        self.rew_buf[:], self.reset_buf[:], task_success_no_collision = compute_dualarm_reach_reward(
            self.reset_buf,
            self.progress_buf,
            actions,
            dist_left,
            dist_right,
            prev_dist_left,
            prev_dist_right,
            obstacle_penalty_sum,
            collision,
            task_success,
            self.prev_task_success,
            self.max_episode_length,
            self.reward_settings["progress_reward_scale"],
            self.reward_settings["distance_penalty_scale"],
            self.reward_settings["reach_bonus_scale"],
            self.reward_settings["lagging_arm_penalty_scale"],
            self.reward_settings["obstacle_penalty_scale"],
            self.reward_settings["collision_penalty"],
            self.reward_settings["success_bonus"],
            self.reward_settings["action_penalty_scale"],
            self.reward_settings["step_penalty"],
        )
        self.prev_task_success = task_success_no_collision.detach()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        num_resets = len(env_ids)
        default_pos = self.default_dof_pos.to(self.device)
        self.dof_pos[env_ids] = default_pos
        self.dof_vel[env_ids] = 0.0
        self.pos_targets[env_ids] = default_pos
        self.effort_targets[env_ids] = 0.0

        self.target_pos[env_ids] = self.target_pos_base.to(self.device).unsqueeze(0)
        self.target_pos[env_ids] += (
            2.0
            * torch.rand((num_resets, self.N_ARMS, 3), device=self.device)
            - 1.0
        ) * self.target_pos_noise.to(self.device)

        self.target_quat[env_ids] = self.target_quat_base.to(self.device).unsqueeze(0)

        self.obstacle_pos[env_ids] = self.obstacle_base_pos.to(self.device).unsqueeze(0)
        self.obstacle_pos[env_ids] += (
            2.0
            * torch.rand((num_resets, self.obstacle_base_pos.shape[0], 3), device=self.device)
            - 1.0
        ) * self.obstacle_pos_noise.to(self.device)

        robot_actor_indices = self.robot_actor_indices[env_ids].to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(robot_actor_indices),
            len(env_ids),
        )

        if self.control_type == "joint_pos":
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.pos_targets),
                gymtorch.unwrap_tensor(robot_actor_indices),
                len(env_ids),
            )
        else:
            self.gym.set_dof_actuation_force_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.effort_targets),
                gymtorch.unwrap_tensor(robot_actor_indices),
                len(env_ids),
            )

        self._refresh()
        ee_pos, ee_quat = self._get_ee_state()
        if self.control_type == "osc":
            self._init_desired_ee_pose(env_ids, ee_pos[env_ids], ee_quat[env_ids])
        self._apply_obstacle_states(env_ids)

        dist = torch.norm(self.target_pos[env_ids] - ee_pos[env_ids], dim=-1)
        self.prev_dist[env_ids] = dist
        self.prev_task_success[env_ids] = False
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        self.compute_observations()
