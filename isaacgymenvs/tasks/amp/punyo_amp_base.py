import glob
import os
import pdb

import numpy as np
#import pickle5 as pickle
import pickle
import torch
from scipy.spatial.transform import Rotation

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    quat_conjugate,
    tensor_clamp,
    to_torch,
    quat_mul,
)
from isaacgymenvs.utils.torch_jit_utils import (
    get_euler_xyz_nowrap,
    quaternion_to_positive_w,
    xyzw_quaternion_to_matrix,
)

from ..base.vec_task import VecTask

from ..punyo_helpers.visualization_helpers import TrajectoryPlotter


def get_obs_size_per_obs_type(obs_type, punyo_num_dofs):
    '''
    Returns the observation size given an observation type.
    '''
    if "robot_dof" == obs_type:
        size = punyo_num_dofs
    elif "robot_vel" == obs_type:
        size = punyo_num_dofs
    elif "box_pose" == obs_type:
        size = 7
    elif "box_vel" == obs_type:
        size = 6
    elif "ee_pose" == obs_type:
        size = 14
    elif "ee_binary_contact" == obs_type:
        size = 2
    elif "floatie_binary_contact" == obs_type:
        size = 14
    elif "env_time" == obs_type:
        size = 1
    elif "previous_actions" == obs_type:
        size = punyo_num_dofs
    elif "target_pos" == obs_type:
        size = 3
    else:
        raise Exception("Unknown observation: {}".format(obs_type))
    return size

def get_obs_size_per_state_type(state_type, punyo_num_dofs):
    '''
    Returns the observation size given a state type.
    The states types are keys of the dictionary self.states.
    '''
    if "dof_pos" == state_type:
        size = punyo_num_dofs
    elif "dof_vel" == state_type:
        size = punyo_num_dofs
    elif "box_pos" == state_type:
        size = 3
    elif "box_quat" == state_type:
        size = 4
    elif "box_vel" == state_type:
        size = 3
    elif "lhand_pos" == state_type:
        size = 3
    elif "lhand_quat" == state_type:
        size = 4
    elif "rhand_pos" == state_type:
        size = 3
    elif "rhand_quat" == state_type:
        size = 4
    elif "paw_pressure" == state_type:
        size = 2
    elif "floatie_pressure" == state_type:
        size = 14
    elif "previous_actions" == state_type:
        size = punyo_num_dofs
    elif "target_pos" == state_type:
        size = 3
    else:
        raise Exception(f"Unknown state type: {state_type}")
    return size


def get_actor_obs_mask(critic_state_types, actor_state_types, punyo_num_dofs):
    # Creates a mask that zeros the privileged observations
    # that are present in the critic_state_types but not in
    # actor_state_types.
    mask_list = []
    for state_type in critic_state_types:
        if state_type in actor_state_types:
            obs_mask = 1
        else:
            obs_mask = 0
        for _ in range(get_obs_size_per_state_type(state_type, punyo_num_dofs)):
            mask_list.append(obs_mask)

    mask = torch.FloatTensor(mask_list)
    return mask

class PunyoAMPBase(VecTask):
    def __init__(
        self,
        config,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
        # The env support actions:
        # - delta_q, delta joint positions.
        # - q, joint positions.
        action_mode="delta_q",
    ):
        self.cfg = config
        self.action_mode = action_mode

        self.dt = self.cfg["sim"]["dt"]
        self.num_dofs = self.cfg["env"]["numActions"]

        self.task_type = self.cfg["task"]["task_type"]
        self.visualize_floaties_contact = self.cfg["env"]["visualizeFloatiesContact"]
        self.plot_trajectories = self.cfg["env"]["plotTrajectories"]
        self.visualize_goal = self.cfg["env"]["visualizeGoal"]

        self.visualize_contact_forces = False
        if (self.cfg["env"]["visualizeContactForces"] and
                not self.cfg["sim"]["use_gpu_pipeline"]):
            self.visualize_contact_forces = self.cfg["env"]["visualizeContactForces"]
        elif (self.cfg["env"]["visualizeContactForces"] and
                self.cfg["sim"]["use_gpu_pipeline"]):
            print("Contact forces can only be visualized "
                    "using cpu pipeline.")

        # Initialize the list of assets to be filled when creating envs.
        self.assets = []

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize_goal = self.cfg["task"]["randomize_goal"]
        if self.randomize_goal:
            self.goal_randomization_params = {
                "xyz_ranges":  [
                    self.cfg["task"]["goal_randomization_params"]["x_range"],
                    self.cfg["task"]["goal_randomization_params"]["y_range"],
                    self.cfg["task"]["goal_randomization_params"]["z_range"],
                ]
            }
            print("\n\tRandomizing the goal position\n")

        # Get the curriculum params.
        self.curriculum = self.cfg["task"]["curriculum"]
        self.curriculum_params = self.cfg["task"]["curriculum_params"]
        self.last_curriculum_steps = None

        # This is a universal action scale applied to all the joints in addition
        # to a joint-wise scaling.
        self.action_scale = self.cfg["env"]["actionScale"]

        # Get robot model-related parameters.
        self.punyo_torque_sensing = self.cfg["env"]["punyo"]["senseTorques"]
        self.punyo_floatie_static_friction = self.cfg["env"]["punyo"][
            "floatieStaticFriction"]
        self.punyo_floatie_torsion_friction = self.cfg["env"]["punyo"][
            "floatieTorsionFriction"]
        self.punyo_torso_static_friction = self.cfg["env"]["punyo"][
            "torsoStaticFriction"]
        self.punyo_torso_torsion_friction = self.cfg["env"]["punyo"][
            "torsoTorsionFriction"]
        self.punyo_paw_mount_static_friction = self.cfg["env"]["punyo"][
            "pawMountStaticFriction"]
        self.punyo_paw_mount_torsion_friction = self.cfg["env"]["punyo"][
            "pawMountTorsionFriction"]
        self.punyo_paw_bubble_static_friction = self.cfg["env"]["punyo"][
            "pawBubbleStaticFriction"]
        self.punyo_paw_bubble_torsion_friction = self.cfg["env"]["punyo"][
            "pawBubbleTorsionFriction"]
        self.punyo_static_friction = self.cfg["env"]["punyo"][
            "staticFriction"]
        self.punyo_torsion_friction = self.cfg["env"]["punyo"][
            "torsionFriction"]
        self.punyo_restitution = self.cfg["env"]["punyo"]["restitution"]
        self.punyo_compliance = self.cfg["env"]["punyo"]["compliance"]

        self.box_static_friction = self.cfg["env"]["object"]["staticFriction"]
        self.box_torsion_friction = self.cfg["env"]["object"]["torsionFriction"]
        self.box_restitution = self.cfg["env"]["object"]["restitution"]
        self.box_compliance = self.cfg["env"]["object"]["compliance"]

        self.table_static_friction = self.cfg["env"]["table"]["staticFriction"]
        self.table_torsion_friction = self.cfg["env"]["table"]["torsionFriction"]
        self.table_restitution = self.cfg["env"]["table"]["restitution"]

        if self.task_type == "task_shelf":
            self.shelf_static_friction = self.cfg["env"]["shelf"]["staticFriction"]
            self.shelf_torsion_friction = self.cfg["env"]["shelf"]["torsionFriction"]
            self.shelf_restitution = self.cfg["env"]["shelf"]["restitution"]

        self.pressure_threshold = self.cfg["env"]["pressureThreshold"]

        self.gravity = self.cfg["sim"]["gravity"]
        up_axis = self.cfg["sim"]["up_axis"]
        if up_axis == "z":
            self.up_axis = gymapi.UP_AXIS_Z
        elif up_axis == "y":
            self.up_axis = gymapi.UP_AXIS_Y
        else:
            assert False, f"up axis type: {up_axis} not implemented."

        # Get the policy (actor) observation types and size.
        self.policy_obs_types = self.cfg["task"][self.task_type]["policyObservation"]
        self.policy_num_obs = self._get_obs_size(self.policy_obs_types)

        if "criticObservation" in self.cfg["task"][self.task_type].keys():
            self.asymmetric_obs = True
            self.critic_obs_types = self.cfg["task"][self.task_type]["criticObservation"]
            # Make sure the policy obs are a subset of the critic obs.
            for policy_obs_type in self.policy_obs_types:
                assert policy_obs_type in self.critic_obs_types, (
                    "the policy obs types should be a subset of the "
                    "critic obs types (privilege obs)")
            self.critic_num_obs = self._get_obs_size(self.critic_obs_types)
        else:
            self.asymmetric_obs = False
            self.critic_num_obs = self.policy_num_obs
            self.critic_obs_types = self.policy_obs_types

        self.collision_mode = self.cfg["env"]["collisionMode"]

        print(f"Critic obs types: {self.critic_obs_types}")
        print(f"Actor obs types: {self.policy_obs_types}")
        # The actor and critic network have the same size corresponding
        # to the critic observation types.
        self.cfg["env"]["numObservations"] = self.critic_num_obs

        self.max_episode_length = self.cfg["task"][self.task_type]["episodeLength"]

        self.motion_dir_path = self.cfg["env"]["asset"]["motion"]["motion_file_path"]
        if self.motion_dir_path == "":
            # If motion file path is not defined try loading from the task.
            self.motion_dir_path = self.cfg["task"][self.task_type]["motion_file_path"]
            print(f"Using TASK motion file path: {self.motion_dir_path}")
        else:
            print(f"Using motion file path: {self.motion_dir_path}")

        motion_files = glob.glob(
            os.path.join(self.motion_dir_path, "**", "*.pkl"), recursive=True
        )
        self.motion_files = []
        for motion_file in motion_files:
            with open(motion_file, "rb") as file:
                self.motion_files.append(pickle.load(file))

        # Get the default goal pose.
        target_x = self.cfg["task"][self.task_type]["target_x"]
        target_y = self.cfg["task"][self.task_type]["target_y"]
        target_z = self.cfg["task"][self.task_type]["target_z"]
        self.default_target_pos = torch.tensor([target_x, target_y, target_z])
        # Get the RPY angles in radians.
        target_yaw = self.cfg["task"][self.task_type]["target_yaw"]
        target_pitch = self.cfg["task"][self.task_type]["target_pitch"]
        target_roll = self.cfg["task"][self.task_type]["target_roll"]
        # Convert the Euler angles to a quaternion.
        target_rot = Rotation.from_euler("xyz", [target_roll, target_pitch, target_yaw])
        self.default_target_quat = torch.tensor(target_rot.as_quat()).float()

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # For success metric calculation.
        self.num_successes_buffer = 0
        self.num_resets_buffer = 0
        self.success_rate = 0
        self.last_success_rate_update = -1

        # get the punyo default poses from the motion file(s).
        self.punyo_default_dof_poses = []

        for motion_file in self.motion_files:
            self.punyo_default_dof_poses.append(
                to_torch(
                    motion_file.iloc[self.cfg["env"]["asset"]["motion"]["ignore_before_idx"]]["observations"]["state"][: self.num_dofs],
                    device=self.device,
                ).unsqueeze(0)
            )
        print(f"\nPunyo default poses:\n{self.punyo_default_dof_poses}\n")
        # Initial actions.
        # Note that it is the same for all the environments which is incorrect
        # because some env might be initialized with a diferent initial configuration.
        #TODO (jose-barreiros): correct the above.
        self._initial_dof_pos = self.punyo_default_dof_poses[0].repeat(self.num_envs, 1)
        self._initial_dof_vel = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device
        )

        # Get the floatie link names.
        floatie_link_names = self.cfg["env"]["punyo"]["floatieLinkNames"]
        # Get the corresponding body handles.
        #TODO (jose-barreiros): The body handles are retrieved for env 0
        # but used for all the enviroments. This is potentially brittle and
        # should be fixed.
        self.punyo_body_handles = {
            "l_paw_mount": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, "paw_mount_link_l"
            ),
            "r_paw_mount": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, "paw_mount_link_l"
            ),
            "l_paw_bubble": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, "paw_bubble_link_l"
            ),
            "r_paw_bubble": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, "paw_bubble_link_r"
            ),
            "l_floatie_0": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[0]
            ),
            "l_floatie_1": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[1]
            ),
            "l_floatie_2": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[2]
            ),
            "l_floatie_3": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[3]
            ),
            "l_floatie_4": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[4]
            ),
            "l_floatie_5": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[5]
            ),
            "l_floatie_6a": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[6]
            ),
            "l_floatie_6b": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[7]
            ),
            "r_floatie_0": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[8]
            ),
            "r_floatie_1": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[9]
            ),
            "r_floatie_2": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[10]
            ),
            "r_floatie_3": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[11]
            ),
            "r_floatie_4": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[12]
            ),
            "r_floatie_5": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[13]
            ),
            "r_floatie_6a": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[14]
            ),
            "r_floatie_6b": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._punyo_actor_id, floatie_link_names[15]
            ),
        }
        # Confirm all the bodies exist in the model.
        for body_name, body_idx in self.punyo_body_handles.items():
            assert body_idx != -1, f"{body_name} does not exist in the model"
        # Get the body indices for the above-defined Punyo bodies.
        self.punyo_body_indices = [*self.punyo_body_handles.values()]
        # Get the body index for the manipuland.
        self.box_body_index = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._box_id, "box")

        # get gym GPU state tensors.
        self._refresh_sim_tensors()

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            self.num_envs, -1,
            13  # 7 pos + 6 vel for the root.
        )
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(
            self.num_envs, -1,
            2  # 1 pos + 1 vel per joint.
        )
        self._dof_force = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_dofs
        )
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(
            self.num_envs, -1,
            3  # Fx, Fy, Fz per body.
        )

        # Initialize the target pose.
        self.target_pos = self.default_target_pos.repeat(self.num_envs, 1).to(self.device)
        self.target_quat = self.default_target_quat.repeat(self.num_envs, 1).to(self.device)

        # Initialize the joint positions and velocities.
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        # Initialize the states for all actors.
        self._initial_root_states = self._root_state.clone()
        self._initial_root_states[:, :, 7:13] = 0  # set velocities to zero.
        self._initial_dof_state = self._dof_state.clone()

        # Initialize the previous actions.
        self._previous_actions = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device
        )

        # Initialize the rigid body states.
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(
            self.num_envs, -1,
            13  # 7 pos + 6 vel for the body.
        )
        self._l_paw_mount_state = self._rigid_body_state[:, self.punyo_body_handles["l_paw_mount"], :]
        self._r_paw_mount_state = self._rigid_body_state[:, self.punyo_body_handles["r_paw_mount"], :]

        self._lpaw_contact = self._contact_forces[:, self.punyo_body_handles["l_paw_bubble"], :]
        self._rpaw_contact = self._contact_forces[:, self.punyo_body_handles["r_paw_bubble"], :]

        self._l_floatie_0 = self._contact_forces[:, self.punyo_body_handles["l_floatie_0"], :]
        self._l_floatie_1 = self._contact_forces[:, self.punyo_body_handles["l_floatie_1"], :]
        self._l_floatie_2 = self._contact_forces[:, self.punyo_body_handles["l_floatie_2"], :]
        self._l_floatie_3 = self._contact_forces[:, self.punyo_body_handles["l_floatie_3"], :]
        self._l_floatie_4 = self._contact_forces[:, self.punyo_body_handles["l_floatie_4"], :]
        self._l_floatie_5 = self._contact_forces[:, self.punyo_body_handles["l_floatie_5"], :]
        self._l_floatie_6a = self._contact_forces[:, self.punyo_body_handles["l_floatie_6a"], :]
        self._l_floatie_6b = self._contact_forces[:, self.punyo_body_handles["l_floatie_6b"], :]

        self._r_floatie_0 = self._contact_forces[:, self.punyo_body_handles["r_floatie_0"], :]
        self._r_floatie_1 = self._contact_forces[:, self.punyo_body_handles["r_floatie_1"], :]
        self._r_floatie_2 = self._contact_forces[:, self.punyo_body_handles["r_floatie_2"], :]
        self._r_floatie_3 = self._contact_forces[:, self.punyo_body_handles["r_floatie_3"], :]
        self._r_floatie_4 = self._contact_forces[:, self.punyo_body_handles["r_floatie_4"], :]
        self._r_floatie_5 = self._contact_forces[:, self.punyo_body_handles["r_floatie_5"], :]
        self._r_floatie_6a = self._contact_forces[:, self.punyo_body_handles["r_floatie_6a"], :]
        self._r_floatie_6b = self._contact_forces[:, self.punyo_body_handles["r_floatie_6b"], :]

        self._box_state = self._root_state[:, self._box_id, :]  # xyzquat

        # Create a goal state reference for visualization.
        if self.visualize_goal:
            self._goal_state = self._root_state[:, self._goal_id, :]

        self._terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # Get the constant params.
        self.workspace_upper_limit_x = self.cfg["task"][self.task_type][
            "workspace_upper_limit_x"
        ]
        self.workspace_lower_limit_x = self.cfg["task"][self.task_type][
            "workspace_lower_limit_x"
        ]
        self.workspace_upper_limit_y = self.cfg["task"][self.task_type][
            "workspace_upper_limit_y"
        ]
        self.workspace_lower_limit_y = self.cfg["task"][self.task_type][
            "workspace_lower_limit_y"
        ]
        self.torque_threshold = self.cfg["task"][self.task_type][
            "torque_threshold"
        ]
        self.contact_force_threshold = self.cfg["task"][self.task_type][
            "contact_force_threshold"
        ]
        self.success_early_termination = self.cfg["task"][self.task_type][
            "success_early_termination"
        ]
        self.translation_reward_w = self.cfg["task"][self.task_type][
            "translation_w"
        ]
        self.keypoint_reward_w = self.cfg["task"][self.task_type][
            "keypoint_w"
        ]
        self.keypoints = self.cfg["task"][self.task_type][
            "keypoints"
        ]
        self.rotation_reward_w = self.cfg["task"][self.task_type][
            "rotation_w"
        ]
        self.action_penalty_w = self.cfg["task"][self.task_type][
            "action_penalty_w"
        ]
        self.box_lin_vel_penalty_w = self.cfg["task"][self.task_type][
            "box_lin_vel_penalty_w"
        ]
        self.box_rot_vel_penalty_w = self.cfg["task"][self.task_type][
            "box_rot_vel_penalty_w"
        ]
        self.torque_penalty_w = self.cfg["task"][self.task_type][
            "torque_penalty_w"
        ]
        self.action_diff_penalty_w = self.cfg["task"][self.task_type][
            "action_diff_penalty_w"
        ]
        self.punyo_contact_penalty_w = self.cfg["task"][self.task_type][
            "punyo_contact_penalty_w"
        ]
        self.box_contact_penalty_w = self.cfg["task"][self.task_type][
            "box_contact_penalty_w"
        ]
        self.success_bonus_w = self.cfg["task"][self.task_type][
            "success_bonus_w"
        ]
        self.goal_xyz_tolerance = self.cfg["task"][self.task_type][
            "goal_xyz_tolerance"
        ]
        self.goal_rot_tolerance = self.cfg["task"][self.task_type][
            "goal_rot_tolerance"
        ]
        self.success_vel = self.cfg["task"][self.task_type][
            "success_vel"
        ]
        self.success_vel_rot = self.cfg["task"][self.task_type][
            "success_vel_rot"
        ]
        self.max_cumulative_successes = self.cfg["task"][self.task_type][
            "max_cumulative_successes"
        ]
        # Initialize variable states.
        self.states = {}
        self._update_states()
        # Add the constants for the task reward into the state.
        self.states.update(
            {
                # Get the workspace limits.
                "workspace_upper_limit_x": torch.tensor(self.workspace_upper_limit_x),
                "workspace_lower_limit_x": torch.tensor(self.workspace_lower_limit_x),
                "workspace_upper_limit_y": torch.tensor(self.workspace_upper_limit_y),
                "workspace_lower_limit_y": torch.tensor(self.workspace_lower_limit_y),
                # Get the torque threshold (for termination condition).
                "torque_threshold": torch.tensor(self.torque_threshold),
                # Get the contact force threshold (for termination condition).
                "contact_force_threshold": torch.tensor(self.contact_force_threshold),
                # Get the flag for success early termination.
                "success_early_termination": torch.tensor(self.success_early_termination),
                # Get the keypoints.
                "keypoints": torch.tensor(self.keypoints).to(self.device),
                # Get the reward weights.
                "keypoint_reward_w": torch.tensor(self.keypoint_reward_w),
                "translation_reward_w": torch.tensor(self.translation_reward_w),
                "rotation_reward_w": torch.tensor(self.rotation_reward_w),
                # Get the success conditions.
                "goal_xyz_tolerance": torch.tensor(self.goal_xyz_tolerance),
                "goal_rot_tolerance": torch.tensor(self.goal_rot_tolerance),
                "success_vel": torch.tensor(self.success_vel),
                "success_vel_rot": torch.tensor(self.success_vel_rot),
                "max_cumulative_successes": torch.tensor(self.max_cumulative_successes),
                # Get the penalty weights.
                "action_penalty_w": torch.tensor(self.action_penalty_w),
                "box_lin_vel_penalty_w": torch.tensor(self.box_lin_vel_penalty_w),
                "box_rot_vel_penalty_w": torch.tensor(self.box_rot_vel_penalty_w),
                "torque_penalty_w": torch.tensor(self.torque_penalty_w),
                "action_diff_penalty_w": torch.tensor(self.action_diff_penalty_w),
                "punyo_contact_penalty_w": torch.tensor(self.punyo_contact_penalty_w),
                "box_contact_penalty_w": torch.tensor(self.box_contact_penalty_w),
                "success_bonus_w": torch.tensor(self.success_bonus_w),
            }
        )

        # Add the goal pose.
        self.states.update(
            {
                "target_pos": self.target_pos,
                "target_quat": quaternion_to_positive_w(self.target_quat),
                "target_rpy": torch.stack(get_euler_xyz_nowrap(self.target_quat), axis=0),
            }
        )

        # Initialize actions.
        self.punyo_dof_pos_targets = torch.clone(self._initial_dof_pos)

        # Initialize cumulative successes buffer.
        self.cumulative_successes = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        # Create the trajectory plotter if opted.
        if self.plot_trajectories:
            self.trajectory_plotter = TrajectoryPlotter(
                dt=self.dt,
                max_episode_length=self.max_episode_length,
                qpos_lb=self.punyo_dof_pos_lb.cpu().numpy(),
                qpos_ub=self.punyo_dof_pos_ub.cpu().numpy(),
                save_data=True,
            )

        self.policy_state_types = self._map_obs_types_to_state_types(self.policy_obs_types)
        self.critic_state_types = self._map_obs_types_to_state_types(self.critic_obs_types)

        if self.asymmetric_obs:
            # Calculate the corresponding actor mask.
            self.actor_mask = get_actor_obs_mask(
                self.critic_state_types,
                self.policy_state_types,
                self.num_dofs).to(rl_device)
            assert self.actor_mask.shape[0] == self.critic_num_obs, "The mask size doesn't match the critic input size"

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def _map_obs_types_to_state_types(self, obs_types):
        # Map observation types to state types .
        obs_ = []
        for p_obs in obs_types:
            if "robot_dof" == p_obs:
                obs_ += ["dof_pos"]
            elif "robot_vel" == p_obs:
                obs_ += ["dof_vel"]
            elif "box_pose" == p_obs:
                obs_ += ["box_pos", "box_quat"]
            elif "box_vel" == p_obs:
                obs_ += ["box_vel"]
            elif "ee_pose" == p_obs:
                obs_ += ["l_paw_mount_pos", "l_paw_mount_quat", "r_paw_mount_pos", "r_paw_mount_quat"]
            elif "ee_binary_contact" == p_obs:
                obs_ += ["paw_pressure"]
            elif "floatie_binary_contact" == p_obs:
                obs_ += ["floatie_pressure"]
            elif "env_time" == p_obs:
                obs_ += ["progress_time"]
            elif "previous_actions" == p_obs:
                obs_ += ["previous_actions"]
            elif "target_pos" == p_obs:
                obs_ += ["target_pos"]
            else:
                raise Exception(f"Unknown observation type: {p_obs}")
        return obs_

    def _get_obs_size(self, obs_types):
        """
        Calculates the observation size given the types.
        """
        num_obs = 0
        for obs in obs_types:
            num_obs += get_obs_size_per_obs_type(obs, self.num_dofs)
        return num_obs

    def create_sim(self):
        self.sim_params.up_axis = self.up_axis
        self.sim_params.gravity.x = self.gravity[0]
        self.sim_params.gravity.y = self.gravity[1]
        self.sim_params.gravity.z = self.gravity[2]
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        # Create environments
        self._create_envs(
            self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # Initialize indices
        num_assets = len(self.assets)
        print(f"\tNumber of assets: {num_assets}")
        self._global_indices = torch.arange(
            self.num_envs * num_assets, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def reset_idx(self, env_ids):
        # Apply randomizations when resetting the envs.
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        # Apply curriculum when resetting the envs.
        if self.curriculum:
            self._apply_curriculum(self.curriculum_params)
        # Randomize the goal when resetting the envs.
        if self.randomize_goal:
            self._apply_goal_randomization(self.goal_randomization_params, env_ids)
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)

    def _edit_asset_props(self, props, static_friction, torsion_friction, restitution, compliance=None, common_filter=None):
        """
        Edit the rigid shape properties of all shapes in props:
        friction, torsion_friction, restitution, and filter.

        Common filter is used to filter out self-collisions
        between all the bodies in props.
        O: enable self-collisions, 1: disable self-collisions
        """
        for p in props:
            p.friction = static_friction
            p.torsion_friction = torsion_friction
            p.restitution = restitution

            if common_filter is not None:
                # Collision filter bitmask - shapes A and B only collide if (filterA & filterB) == 0
                p.filter = common_filter
            if compliance is not None:
                p.compliance = compliance
        return props

    def _construct_pose(self, x, y, z, q_x, q_y, q_z, q_w):
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(
            x, y, z)
        start_pose.r = gymapi.Quat(
            q_x, q_y, q_z, q_w)
        return start_pose

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"].get("assetRoot")
        punyo_asset_file = self.cfg["env"]["asset"].get("assetFilePunyo")
        # load punyo asset.
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = False  # this only affects the visual.
        punyo_asset = self.gym.load_asset(
            self.sim, asset_root, punyo_asset_file, asset_options
        )
        # Add to the list of assets.
        self.assets.append(punyo_asset)

        # Set the joint stiffness.
        if self.cfg["env"]["punyo"].get("waistStiffness"):
            joint_stiffness = np.concatenate((
                self.cfg["env"]["punyo"].get("waistStiffness"),
                np.tile(self.cfg["env"]["punyo"].get("jointStiffness"), 2)
            ))
        else:
            joint_stiffness = np.tile(
                self.cfg["env"]["punyo"].get("jointStiffness"), 2
            )
        assert len(joint_stiffness) == self.num_dofs, \
            "The joint stiffness must have the same size with the DOF."
        punyo_dof_stiffness = to_torch(
            joint_stiffness, dtype=torch.float, device=self.device,
        )
        # Set the joint damping.
        if self.cfg["env"]["punyo"].get("waistDamping"):
            joint_damping = np.concatenate((
                self.cfg["env"]["punyo"].get("waistDamping"),
                np.tile(self.cfg["env"]["punyo"].get("jointDamping"), 2)
            ))
        else:
            joint_damping = np.tile(
                self.cfg["env"]["punyo"].get("jointDamping"), 2
            )
        assert len(joint_damping) == self.num_dofs, \
            "The joint damping must have the same size with the DOF."
        punyo_dof_damping = to_torch(
            joint_damping, dtype=torch.float, device=self.device
        )
        # Set the joint speed limits.
        if self.cfg["env"]["punyo"].get("waistSpeedLimit"):
            joint_speed_limit = np.concatenate((
                self.cfg["env"]["punyo"].get("waistSpeedLimit"),
                np.tile(self.cfg["env"]["punyo"].get("jointSpeedLimit"), 2)
            ))
        else:
            joint_speed_limit = np.tile(
                self.cfg["env"]["punyo"].get("jointSpeedLimit"), 2
            )
        assert len(joint_speed_limit) == self.num_dofs, \
            "The joint speed limit must have the same size with the DOF."
        self.punyo_dof_speed_lim = to_torch(
            joint_speed_limit, dtype=torch.float, device=self.device
        )

        # Create table asset.
        TABLE_THICKNESS = 0.05
        TABLE_WIDTH = 1.8
        TABLE_LENGTH = 1.8
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, TABLE_WIDTH, TABLE_LENGTH, TABLE_THICKNESS, table_opts
        )
        # Add to the list of assets.
        self.assets.append(table_asset)

        # Create shelf asset
        if self.task_type == "task_shelf":
            shelf_asset_file = self.cfg["task"][self.task_type].get("assetFileShelf")
            shelf_opts = gymapi.AssetOptions()
            shelf_opts.fix_base_link = True
            shelf_asset = self.gym.load_asset(self.sim, asset_root, shelf_asset_file, shelf_opts)
            # Add to the list of assets.
            self.assets.append(shelf_asset)

        # Create a triad to visualize the goal.
        if self.visualize_goal:
            goal_opts = gymapi.AssetOptions()
            goal_opts.fix_base_link = True
            goal_asset_file = "urdf/objects/goal_triad.urdf"
            goal_asset = self.gym.load_asset(self.sim, asset_root, goal_asset_file, goal_opts)
            # Add to the list of assets.
            self.assets.append(goal_asset)

        # Create box asset.
        box_opts = gymapi.AssetOptions()
        box_asset_file = self.cfg["task"][self.task_type].get("assetFileBox")
        box_asset = self.gym.load_asset(self.sim, asset_root, box_asset_file, box_opts)
        # Add to the list of assets.
        self.assets.append(box_asset)

        # Get the number of degrees of freedom for Punyo.
        self.num_punyo_dofs = self.gym.get_asset_dof_count(punyo_asset)
        print("\tNumber of DOF of Punyo: ", self.num_punyo_dofs)

        # set punyo dof properties.
        punyo_dof_props = self.gym.get_asset_dof_properties(punyo_asset)
        self.punyo_dof_pos_lb = []
        self.punyo_dof_pos_ub = []
        for i in range(self.num_punyo_dofs):
            punyo_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                # TODO: Consider setting the user-specified speed limits here as well.
                punyo_dof_props["stiffness"][i] = punyo_dof_stiffness[i]
                punyo_dof_props["damping"][i] = punyo_dof_damping[i]

            self.punyo_dof_pos_lb.append(punyo_dof_props["lower"][i])
            self.punyo_dof_pos_ub.append(punyo_dof_props["upper"][i])

        self.punyo_dof_pos_lb = to_torch(
            self.punyo_dof_pos_lb, device=self.device
        )
        self.punyo_dof_pos_ub = to_torch(
            self.punyo_dof_pos_ub, device=self.device
        )

        punyo_props = self.gym.get_asset_rigid_shape_properties(punyo_asset)

        if self.collision_mode == 2:
            # allow all self-collisions (including floaties)
            common_filter = 0
        elif self.collision_mode == 1:
            # disable all self-collisions
            common_filter = 1
        elif self.collision_mode == 0:
            common_filter = None
        else:
            raise NotImplementedError
        punyo_props = self._edit_asset_props(
            punyo_props, self.punyo_static_friction,
            self.punyo_torsion_friction, self.punyo_restitution,
            compliance=self.punyo_compliance,
            common_filter=common_filter
            )

        # Get the body index - body name and body index - shape index maps.
        map_body_name_to_body_index = \
            self.gym.get_asset_rigid_body_dict(punyo_asset)
        map_body_index_to_shape_index_range = \
            self.gym.get_asset_rigid_body_shape_indices(punyo_asset)
        # Get the floatie and torso shapes.
        floatie_shape_indices = []
        torso_shape_indices = []
        for key in map_body_name_to_body_index.keys():
            shape_start = map_body_index_to_shape_index_range[
                map_body_name_to_body_index[key]].start
            shape_count = map_body_index_to_shape_index_range[
                map_body_name_to_body_index[key]].count
            if "floatie" in key:
                floatie_shape_indices.append(shape_start)
            if "torso" in key:
                torso_shape_indices.append(shape_start)
            # Make sure there is only one shape in the floatie/torso bodies.
            if ("floatie" in key) or ("torso" in key):
                assert shape_count == 1, ("There seems to be more than one "
                                         f"collision geom in the {key} body, "
                                          "hence you should take extra steps "
                                          "to set the friction properly.")
        # Get the paw shapes.
        paw_mount_shape_indices = {}
        paw_bubble_shape_indices = {}
        for side_id, side_name in enumerate(["l", "r"]):
            paw_mount_shape_indices[side_name] = []
            # Get the mount shape indices.
            mount_body_name = "paw_mount_link_" + side_name
            paw_mount_shape_start = map_body_index_to_shape_index_range[
                map_body_name_to_body_index[mount_body_name]].start
            paw_mount_num_shapes = map_body_index_to_shape_index_range[
                map_body_name_to_body_index[mount_body_name]].count
            for shape_i in range(paw_mount_num_shapes):
                paw_mount_shape_indices[side_name].append(paw_mount_shape_start + shape_i)
            # Get the bubble shape index.
            paw_bubble_shape_indices[side_name] = map_body_index_to_shape_index_range[
                map_body_name_to_body_index["paw_bubble_link_" + side_name]].start

        # Set the collision filters.
        if self.collision_mode == 0:
            # Allow arm but no floatie-floatie nor floatie-torso self-collisions.
            # Filter collisions between floaties.
            # Only shapes with the same non-zero filter value don't collide.
            filters = [0] * len(punyo_props)  # initialize filters with zero to enable self collisions.
            # Get list of body names to be disabled from self-collisions.
            no_collision_body_names = []
            for key in map_body_name_to_body_index.keys():
                if ("floatie" in key) or ("torso" in key):
                    no_collision_body_names.append(key)
            # This filters collisions between the 1.0-floaties and the corresponding links.
            # The unintended effect is that self collisions between Links 1, 3, 4, 5, and 6
            # in both arms are disabled.
            no_collision_body_names += self.cfg["env"]["punyo"]["noCollisionBodyNames"]
            # Set the shape filter to the same value to disable collisions
            # between the floaties.
            for body_name in no_collision_body_names:
                shape_index = map_body_index_to_shape_index_range[
                    map_body_name_to_body_index[body_name]].start
                filters[shape_index] = 100  # this values does not matter as long as it is not 0.
            # Filter out self collisions within each paw while allowing
            # collisions between the paws.
            for side_id, side_name in enumerate(["l", "r"]):
                # NOTE: In theory, this is supposed to be set for all the shapes
                # in the paw body, i.e., not only for the first geometry in the
                # mount link. However, doing so causes all collisions between
                # the arms to be ignored. Hence, we are setting the collision
                # filter for only the first collision geom in the paw mount.
                filters[paw_mount_shape_indices[side_name][0]] = 200 + side_id
                filters[paw_bubble_shape_indices[side_name]] = 200 + side_id
            # Set the shape filters.
            for i, p in enumerate(punyo_props):
                p.filter = filters[i]

        # Modify the torso frictions.
        for shape in torso_shape_indices:
            punyo_props[shape].friction = self.punyo_torso_static_friction
            punyo_props[shape].torsion_friction = self.punyo_torso_torsion_friction
        # Modify the floatie frictions.
        for shape in floatie_shape_indices:
            punyo_props[shape].friction = self.punyo_floatie_static_friction
            punyo_props[shape].torsion_friction = self.punyo_floatie_torsion_friction
        # Modify the paw mount frictions.
        for shape in np.hstack(paw_mount_shape_indices.values()):
            punyo_props[shape].friction = self.punyo_paw_mount_static_friction
            punyo_props[shape].torsion_friction = self.punyo_paw_mount_torsion_friction
        # Modify the paw bubble frictions.
        for shape in np.hstack(paw_bubble_shape_indices.values()):
            punyo_props[shape].friction = self.punyo_paw_bubble_static_friction
            punyo_props[shape].torsion_friction = self.punyo_paw_bubble_torsion_friction

        # Set the shape properties for Punyo.
        self.gym.set_asset_rigid_shape_properties(punyo_asset, punyo_props)

        # Set the shape properties for the other assets in the environment.
        box_props = self.gym.get_asset_rigid_shape_properties(box_asset)
        box_props = self._edit_asset_props(
            box_props, self.box_static_friction,
            self.box_torsion_friction, self.box_restitution, compliance=self.box_compliance)
        self.gym.set_asset_rigid_shape_properties(box_asset, box_props)

        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        table_props = self._edit_asset_props(
            table_props, self.table_static_friction,
            self.table_torsion_friction, self.table_restitution)
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)

        if self.task_type == "task_shelf":
            shelf_props = self.gym.get_asset_rigid_shape_properties(shelf_asset)
            shelf_props = self._edit_asset_props(
                shelf_props, self.shelf_static_friction,
                self.shelf_torsion_friction, self.shelf_restitution)
            self.gym.set_asset_rigid_shape_properties(shelf_asset, shelf_props)

        # Define start pose for punyo
        punyo_start_pose = self._construct_pose(
            self.cfg["env"]["punyo"]["init_x"],
            self.cfg["env"]["punyo"]["init_y"],
            self.cfg["env"]["punyo"]["init_z"],
            self.cfg["env"]["punyo"]["init_q_x"],
            self.cfg["env"]["punyo"]["init_q_y"],
            self.cfg["env"]["punyo"]["init_q_z"],
            self.cfg["env"]["punyo"]["init_q_w"],
        )

        # Define start pose for table
        table_start_pose = self._construct_pose(
            self.cfg["env"]["table"]["init_x"],
            self.cfg["env"]["table"]["init_y"],
            self.cfg["env"]["table"]["init_z"] - TABLE_THICKNESS / 2,
            self.cfg["env"]["table"]["init_q_x"],
            self.cfg["env"]["table"]["init_q_y"],
            self.cfg["env"]["table"]["init_q_z"],
            self.cfg["env"]["table"]["init_q_w"],
        )

        # Define start pose for the shelf for the shelving task
        if self.task_type == "task_shelf":
            shelf_start_pose = self._construct_pose(
                self.cfg["env"]["shelf"]["init_x"],
                self.cfg["env"]["shelf"]["init_y"],
                self.cfg["env"]["shelf"]["init_z"],
                self.cfg["env"]["shelf"]["init_q_x"],
                self.cfg["env"]["shelf"]["init_q_y"],
                self.cfg["env"]["shelf"]["init_q_z"],
                self.cfg["env"]["shelf"]["init_q_w"],
            )

        # Define the default goal pose for visualization.
        if self.visualize_goal:
            goal_start_pose = self._construct_pose(
                self.default_target_pos[0],
                self.default_target_pos[1],
                self.default_target_pos[2],
                self.default_target_quat[0],
                self.default_target_quat[1],
                self.default_target_quat[2],
                self.default_target_quat[3],
            )

        # Define a dummy start pose for box.
        # NOTE: These values don't really matter since they are overwritten
        # during the reset anyways.
        box_start_pose = self._construct_pose(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        # Compute aggregate size.
        max_agg_bodies = 0
        max_agg_shapes = 0
        for asset_id, asset in enumerate(self.assets):
            # Check for the existence of the asset.
            num_asset_bodies = self.gym.get_asset_rigid_body_count(asset)
            num_asset_shapes = self.gym.get_asset_rigid_shape_count(asset)
            print(f"\tAsset #{asset_id} has {num_asset_bodies} bodies and {num_asset_shapes} shapes")
            # Accumulate the maximum number of bodies and shapes.
            max_agg_bodies += num_asset_bodies
            max_agg_shapes += num_asset_shapes

        # Create environments.
        self.envs = []
        for i in range(self.num_envs):
            # create env instance.
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Aggregate all the actors together.
            # NOTE: An aggregate is a collection of actors. Aggregates do not provide extra simulation functionality,
            # but allow you to tell PhysX that a set of actors will be clustered together, which in
            # turn allows PhysX to optimize its spatial data operations. It is not necessary to create
            # aggregates, but doing so can provide a modest performance boost. Please see the
            # following doc for more detailts:
            # https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/guide/Manual/RigidBodyCollision.html#aggregates
            # NOTE: Currently, all the assets in an environment are aggregated together.
            # However, a more modularized version of this is available in commit d927562ec900a0c7fa603bf4a3ff2c658377df0b.
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # NOTE: punyo should ALWAYS be loaded first in sim! This is an inherited style from other Isaac envs and we don't know the reason why.
            # Create punyo
            # 0: allow self-collisions, -1: otherwise (aka. predefined collisions).
            # Predefined collisions default to whatever is used in the set_asset_rigid_shape_properties()
            # https://forums.developer.nvidia.com/t/self-collision-avoiding-parent-links-or-links-higher-in-the-kinematic-chain/196626/6
            self._punyo_actor_id = self.gym.create_actor(
                env_ptr, punyo_asset, punyo_start_pose, "punyo", i
            )
            assert self._punyo_actor_id == 0, "Punyo handle should always be zero."
            self.gym.set_actor_dof_properties(env_ptr, self._punyo_actor_id, punyo_dof_props)
            # Enable joint force sensors.
            if self.punyo_torque_sensing:
                self.gym.enable_actor_dof_force_sensors(env_ptr, self._punyo_actor_id)

            # Create table
            table_actor_id = self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i
            )
            # Set the color
            table_color = gymapi.Vec3(0.85, 0.85, 0.7)  # RGB
            self.gym.set_rigid_body_color(
                env_ptr, table_actor_id, 0, gymapi.MESH_VISUAL, table_color
            )

            # Create shelf for the box shelving task.
            if self.task_type == "task_shelf":
                shelf_actor_id = self.gym.create_actor(
                    env_ptr, shelf_asset, shelf_start_pose, "shelf", i
                )

            # Create goal actor.
            if self.visualize_goal:
                self._goal_id = self.gym.create_actor(
                    env_ptr, goal_asset, goal_start_pose, "goal", -1
                )

            # Create box
            self._box_id = self.gym.create_actor(
                env_ptr, box_asset, box_start_pose, "object", i
            )
            if i == 0:
                box_id = self._box_id
            assert self._box_id == box_id, "self._box_id should be the same across envs."
            box_id = self._box_id
            # Set the color
            box_color = gymapi.Vec3(0.6, 0.1, 0.0)  # RGB
            self.gym.set_rigid_body_color(
                env_ptr, self._box_id, 0, gymapi.MESH_VISUAL, box_color
            )

            # End aggregation.
            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers.
            self.envs.append(env_ptr)

        # Setup init state buffer
        self.box_default_poses = []
        for motion_file in self.motion_files:
            # Use manipuland's pose from the first frame of the motion file as the default pose
            # box_pose = motion_file.iloc[0]["observations"]["manipuland_rpyxyz"]

            # Define the default pose rpyxyz for the box at the center of the table
            box_pose = [
                self.cfg["task"][self.task_type]["default_roll"],
                self.cfg["task"][self.task_type]["default_pitch"],
                self.cfg["task"][self.task_type]["default_yaw"],
                self.cfg["task"][self.task_type]["default_x"],
                self.cfg["task"][self.task_type]["default_y"],
                self.cfg["task"][self.task_type]["default_z"],
                ]
            rot = Rotation.from_euler("xyz", box_pose[0:3], degrees=False)
            rot_quat = rot.as_quat()
            self.box_default_poses.append(
                to_torch(
                    np.concatenate((box_pose[3:6], rot_quat, np.zeros(6))),
                    device=self.device,
                ).unsqueeze(0)
            )

    def _apply_curriculum(self, curriculum_params):
        assert "reward_weights" in curriculum_params, "Only supports reward weights curriculum."

        if "reward_weights" in curriculum_params:
            reward_weights_params = curriculum_params["reward_weights"]
            if self.last_curriculum_steps is None:
                # Init the variable that keeps track of the last step
                # a curriculum action was applied.
                self.last_curriculum_steps = [-1] * len(reward_weights_params)

            for i, rew_weight_name in enumerate(reward_weights_params):
                weight_params = reward_weights_params[rew_weight_name]
                mode = weight_params["mode"]
                operation = weight_params["operation"]
                direction = weight_params["direction"]
                value = weight_params["value"]

                if mode == "fixed":
                    # Fixed curriculum that is triggered every frequency steps and
                    # starts at and ends at start_at and stop_at steps
                    # respectively.
                    frequency = weight_params["mode_params"]["frequency"]
                    start_at = weight_params["mode_params"]["start_at"]
                    stop_at = weight_params["mode_params"]["stop_at"]
                    trigger_condition = (
                        self.last_step - self.last_curriculum_steps[i] > frequency and
                        self.last_step > start_at and
                        self.last_step < stop_at)
                elif mode == "automatic":
                    # Automatic curriculum that is triggered when success rate exceeds
                    # the success_rate_threshold and for a minimum of minimum_steps_per_curriculum steps.
                    success_rate_threshold = weight_params["mode_params"]["success_rate_threshold"]
                    minimum_steps_per_curriculum = weight_params["mode_params"]["minimum_steps_per_curriculum"]
                    trigger_condition = (
                        self.success_rate > success_rate_threshold and
                        self.last_step - self.last_curriculum_steps[i] > minimum_steps_per_curriculum
                    )
                else:
                    assert False, f"mode: {mode} not supported."

                if trigger_condition:
                    # Apply curriculum.
                    old_value = getattr(vars()["self"], rew_weight_name)
                    if operation == "additive":
                        step_value = value
                    elif operation == "exponential":
                        step_value = old_value * value
                    else:
                        assert False, f"Operation {operation} not suported"

                    if direction == "increase":
                        new_value =  old_value + step_value
                    elif direction == "decrease":
                        new_value = max(
                            0,
                            old_value - step_value)
                    else:
                        assert False, f"Direction {direction} not suported"
                    setattr(vars()["self"], rew_weight_name, new_value)
                    if i == 0:
                        print("\n")
                    print(f"Curriculum:  '{rew_weight_name}' changed from {old_value} to {new_value} "
                          f"in step {self.last_step}")

                    # Update the state.
                    self.states.update(
                        {
                        rew_weight_name: torch.tensor(getattr(vars()["self"], rew_weight_name)),
                        })
                    self.last_curriculum_steps[i] = self.last_step

    def _apply_goal_randomization(self, goal_randomization_params, env_ids):
        "Apply additive uniform noise to the goal position."
        # Get the perturbation range values.
        rand_xyz_ranges = goal_randomization_params["xyz_ranges"]
        # Apply +/- perturbations to the horizontal position.
        self.target_pos[env_ids, 0:1] = self.default_target_pos[0] + (
            rand_xyz_ranges[0] * 2.0 * (torch.rand(len(env_ids), 1, device=self.device) - 0.5)
        )
        self.target_pos[env_ids, 1:2] = self.default_target_pos[1] + (
            rand_xyz_ranges[1] * 2.0 * (torch.rand(len(env_ids), 1, device=self.device) - 0.5)
        )
        # For the height perturb only in the positive direction.
        self.target_pos[env_ids, 2:3] = self.default_target_pos[2] + (
            rand_xyz_ranges[2] * torch.rand(len(env_ids), 1, device=self.device)
        )
        # Update the goal pose for visualization.
        if self.visualize_goal:
            self._goal_state[env_ids, 0:3] = self.target_pos[env_ids, :]
            self._goal_state[env_ids, 3:7] = self.target_quat[env_ids, :]
        # Update the state dict.
        self.states.update({"target_pos": self.target_pos})


    def _compute_reward(self, actions, previous_actions):
        old_num_resets = torch.sum(self.reset_buf.float())

        (self.rew_buf[:], self.reset_buf[:], individual_rewards,
         successes, success_metrics, self.cumulative_successes[:]) = compute_task_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            actions,
            previous_actions,
            self.states,
            self.max_episode_length,
            self.cumulative_successes,
        )

        # Log the mean cumulative individual reward components among all environments.
        # Note that the mapping of individual_rewards must correspond to
        # the ordering of compute_task_reward().
        episode_cumulative = dict()
        episode_cumulative["dist_rew"] = individual_rewards[0]
        episode_cumulative["rot_rew"] = individual_rewards[1]
        episode_cumulative["keypoint_rew"] = individual_rewards[2]
        episode_cumulative["action_penalty"] = individual_rewards[3]
        episode_cumulative["action_diff_penalty"] = individual_rewards[4]
        episode_cumulative["box_lin_vel_penalty"] = individual_rewards[5]
        episode_cumulative["box_rot_vel_penalty"] = individual_rewards[6]
        episode_cumulative["torque_penalty"] = individual_rewards[7]
        episode_cumulative["punyo_contact_penalty"] = individual_rewards[8]
        episode_cumulative["box_contact_penalty"] = individual_rewards[9]
        episode_cumulative["drop_penalty"] = individual_rewards[10]
        episode_cumulative["out_of_bound_penalty"] = individual_rewards[11]
        episode_cumulative["success_bonus"] = individual_rewards[12]
        self.extras['episode_cumulative'] = episode_cumulative
        # Log the absolute max velocity among all joints across all environments.
        episode_direct_info = dict()
        episode_direct_info["max_abs_joint_vel"] = torch.max(torch.max(torch.abs(self.states["dof_vel"]), dim=1).values)
        episode_direct_info["max_abs_joint_torque"] = torch.max(torch.max(torch.abs(self.states["dof_force"]), dim=1).values)
        # Log the max contact force norms among all the bodies of the robot across all environments.
        episode_direct_info["max_punyo_contact_force"] = torch.max(
            torch.max(
                torch.norm(
                    self.states["punyo_contact_forces"], p=2, dim=2), dim=1).values)
        # Log the curriculum reward weights.
        if self.curriculum:
            for curriculum_w in self.curriculum_params["reward_weights"]:
                episode_direct_info[curriculum_w] = self.states[curriculum_w]
        self.extras['episode_info'] = episode_direct_info
        # Log the success rate.
        episode_success_info = dict()
        episode_success_info["goal_distance"] = torch.mean(success_metrics[0])
        episode_success_info["rot_distance"] = torch.mean(success_metrics[1])
        episode_success_info["manipuland_vel"] = torch.mean(success_metrics[2])
        episode_success_info["manipuland_vel_rot"] = torch.mean(success_metrics[3])
        # Success rate is defined as the cumulative number of successes divided
        # by the cumulative number of resets in a window of WINDOW_SIZE episodes.
        WINDOW_SIZE = 1  # num of episodes for which the success_rate is calculated.
        update_freq = (self.max_episode_length - 1) * WINDOW_SIZE * self.num_envs  # number of steps per environment for which we calculate the success rate.
        num_resets = torch.sum(self.reset_buf.float())
        self.num_successes_buffer += torch.sum(successes.float())
        num_new_resets = num_resets - old_num_resets
        self.num_resets_buffer += num_new_resets
        last_step = self.gym.get_frame_count(self.sim)*self.num_envs
        if (last_step - self.last_success_rate_update > update_freq and
            self.num_resets_buffer>0):
            assert self.num_successes_buffer>=0, "Number of successes must not be negative."
            assert self.num_resets_buffer>0, "Number of resets must not be negative or zero."
            # Update success rate at the update frequency given at least
            # one environment has reset.
            # The success rate is clamped to a max of 1 because without success early termination,
            # the number of successes can exceed the number of resets.
            self.success_rate = torch.clamp(self.num_successes_buffer / self.num_resets_buffer, max=1.0)
            self.num_successes_buffer = 0
            self.num_resets_buffer = 0
            self.last_success_rate_update = last_step
            print(f"Success rate updated to: {self.success_rate}")

        episode_success_info["success_rate"] = self.success_rate
        self.extras["success_info"] = episode_success_info

    def _refresh_sim_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _compute_observations(self, env_ids=None):
        # Refresh states.
        self._update_states()
        obs = self._compute_obs_by_type(self.critic_state_types, env_ids)
        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

    def _update_states(self):
        paw_pressures = torch.stack((
            self._lpaw_contact,
            self._rpaw_contact), dim=1)

        paw_pressure_thresholded = torch.norm(paw_pressures, dim=2) > self.pressure_threshold
        paw_binary_contact = paw_pressure_thresholded * 1  # convert binary to float.

        floatie_pressures = torch.stack((
            self._l_floatie_0,
            self._l_floatie_1,
            self._l_floatie_2,
            self._l_floatie_3,
            self._l_floatie_4,
            self._l_floatie_5,
            (self._l_floatie_6a + self._l_floatie_6b)/2,
            self._r_floatie_0,
            self._r_floatie_1,
            self._r_floatie_2,
            self._r_floatie_3,
            self._r_floatie_4,
            self._r_floatie_5,
            (self._r_floatie_6a + self._r_floatie_6b)/2,
            ), dim=1)

        floatie_pressures_thresholded = torch.norm(floatie_pressures, dim=2)  > self.pressure_threshold
        floatie_binary_contact = floatie_pressures_thresholded * 1  # convert binary to float.

        self.states.update(
            {
                # punyo
                "dof_pos": self._dof_pos[:, :],
                "dof_vel": self._dof_vel[:, :],
                "dof_force": self._dof_force[:, :],
                "previous_actions": self._previous_actions[:, :],
                "l_paw_mount_state": self._l_paw_mount_state[:, :7],
                "r_paw_mount_state": self._r_paw_mount_state[:, :7],
                "l_paw_mount_pos": self._l_paw_mount_state[:, :3],
                "l_paw_mount_quat": quaternion_to_positive_w(self._l_paw_mount_state[:, 3:7]),
                "r_paw_mount_pos": self._r_paw_mount_state[:, :3],
                "r_paw_mount_quat": quaternion_to_positive_w(self._r_paw_mount_state[:, 3:7]),
                "progress_time": self.progress_buf[:].reshape(-1,1),
                "punyo_contact_forces": self._contact_forces[:, self.punyo_body_indices],
                "box_contact_forces": self._contact_forces[:, self.box_body_index],
                "l_floatie_0": self._l_floatie_0[:, :],
                "l_floatie_1": self._l_floatie_1[:, :],
                "l_floatie_2": self._l_floatie_2[:, :],
                "l_floatie_3": self._l_floatie_3[:, :],
                "l_floatie_4": self._l_floatie_4[:, :],
                "l_floatie_5": self._l_floatie_5[:, :],
                "l_floatie_6": (self._l_floatie_6a[:, :] + self._l_floatie_6b[:, :])/2,
                "r_floatie_0": self._r_floatie_0[:, :],
                "r_floatie_1": self._r_floatie_1[:, :],
                "r_floatie_2": self._r_floatie_2[:, :],
                "r_floatie_3": self._r_floatie_3[:, :],
                "r_floatie_4": self._r_floatie_4[:, :],
                "r_floatie_5": self._r_floatie_5[:, :],
                "r_floatie_6": (self._r_floatie_6a[:, :] + self._r_floatie_6b[:, :])/2,
                "paw_pressure": paw_binary_contact,
                "floatie_pressure": floatie_binary_contact,
                # box
                "box_quat": quaternion_to_positive_w(self._box_state[:, 3:7]),
                "box_pos": self._box_state[:, :3],
                "box_rpy": torch.stack(get_euler_xyz_nowrap(self._box_state[:, 3:7]), axis=0),
                "box_vel": self._box_state[:, 7:],
            }
        )

    def policy_obs_to_dict(self, obs):
        """
        Creates a dictionary given an obs tensor.
        The values are numpy arrays.
        """
        obs_dict = {}
        i = 0
        for key in self.policy_obs_types:
            j = get_obs_size_per_obs_type(key, self.num_dofs)
            obs_dict[key] = obs[:, i:i+j].to("cpu").numpy()
            i += j
        return obs_dict

    def _compute_obs_by_type(self, state_types, env_ids=None):
        """
        Returns the observations given a set of state types.
        """

        if env_ids is None:
            obs = torch.cat([self.states[ob] for ob in state_types], dim=-1)
        else:
            obs = torch.cat([self.states[ob][env_ids] for ob in state_types], dim=-1)
        return obs

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if self.action_mode == "delta_q":
            # Unnormalize and scale the actions.
            # NOTE: This assumes the actions are normalized (i.e., in [-1, 1]),
            # so they are unnormalizad by the joint speed limits.
            # TODO: Confirm the actions are bounded in [-1-tol, 1+tol].
            actions_scaled = self.action_scale * self.punyo_dof_speed_lim * self.actions
            # Apply the change of joint positions factoring in the time step size.
            # NOTE: delta_q is added to the previous commanded joint positions
            # and not to the previous joint positions.
            targets = (
                self.punyo_dof_pos_targets[:, : self.num_punyo_dofs]
                + actions_scaled * self.dt
            )
            # Make sure the joint position commands satisfy the limits.
            self.punyo_dof_pos_targets[:, : self.num_punyo_dofs] = tensor_clamp(
                targets, self.punyo_dof_pos_lb, self.punyo_dof_pos_ub
            )
        elif self.action_mode == "q":
            # The action joint position is sent directly as targets to the
            # joint controllers.
            # NOTE: This mode is not intended for training.
            targets = self.actions
            self.punyo_dof_pos_targets[:, : self.num_punyo_dofs] = targets
        else:
            assert False, f"Action mode: {self.action_mode} does not exist"

        # Set the command.
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.punyo_dof_pos_targets)
        )

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.extras["terminate"] = self._terminate_buf
        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions, self._previous_actions)
        self._previous_actions = self.actions

        # visualize contacts
        if self.visualize_contact_forces:
            self.gym.clear_lines(self.viewer)
            VECTOR_SCALE = 0.25
            for env in self.envs:
                self.gym.draw_env_rigid_contacts(
                    self.viewer, env,
                    gymapi.Vec3(0.8, 0.8, 0.1),  # rgb
                    VECTOR_SCALE,
                    True)

        # Visualize trajectories.
        if self.plot_trajectories:
            # Get the state of the last environment since that is
            # typically the closest to the viewpoint.
            qpos = self.states["dof_pos"][-1].cpu().numpy()
            qvel = self.states["dof_vel"][-1].cpu().numpy()
            tau = self.states["dof_force"][-1].cpu().numpy()
            qpos_cmd = self.punyo_dof_pos_targets[-1].cpu().numpy()
            # Get the maximum speed per joint across all environments.
            qvel_max = torch.max(torch.abs(self.states["dof_vel"]), axis=0)[0].cpu().numpy()
            # Get the contact forces.
            NUM_FLOATIE_TWOS_PER_ARM = 6
            NUM_CONTACT_BODIES_PER_ARM = (NUM_FLOATIE_TWOS_PER_ARM +
                                          2 +  # for the floatie 1.0 parts
                                          1)   # for the paw
            fc = np.zeros(2 * NUM_CONTACT_BODIES_PER_ARM)
            for side_idx, side in enumerate(["l", "r"]):
                # Get the contact force magnitudes for the floatie 2.0s.
                for floatie_idx in range(NUM_FLOATIE_TWOS_PER_ARM):
                    body_idx = side_idx * NUM_CONTACT_BODIES_PER_ARM + floatie_idx
                    fc[body_idx] = torch.norm(
                        self._contact_forces[:, self.punyo_body_handles[side + '_floatie_' + str(floatie_idx)], :],
                        dim=-1)[-1].cpu().numpy()
                # Get the contact force magnitude for each part of the floatie 1.0.
                for part_id, part_name in enumerate(["a", "b"]):
                    body_idx = (side_idx * NUM_CONTACT_BODIES_PER_ARM +
                                NUM_FLOATIE_TWOS_PER_ARM +
                                part_id)
                    fc[body_idx] = torch.norm(
                        self._contact_forces[:, self.punyo_body_handles[side + '_floatie_6' + part_name], :],
                        dim=-1)[-1].cpu().numpy()
                # Get the contact force magnitude for the paw.
                body_idx = side_idx * NUM_CONTACT_BODIES_PER_ARM + NUM_CONTACT_BODIES_PER_ARM - 1
                fc[body_idx] = torch.norm(
                    self._contact_forces[:, self.punyo_body_handles[side + '_paw_bubble'], :],
                    dim=-1)[-1].cpu().numpy()
            # Pass to the plotter.
            self.trajectory_plotter.plot(
                qpos_i=qpos,
                qvel_i=qvel,
                tau_i=tau,
                qpos_cmd_i=qpos_cmd,
                qvel_max_i=qvel_max,
                fc_i=fc,
            )

        if self.visualize_floaties_contact:
            # Visualize the binary contact states of the floaties and paws
            TOTAL_NUM_FLOATIES_PER_ARM = 7
            NUM_FLOATIE_TWOS_PER_ARM = 6
            for i in range(self.num_envs):
                for side_idx, side in enumerate(["l", "r"]):
                    # Floatie 2.0
                    for j in range(NUM_FLOATIE_TWOS_PER_ARM):
                        self.gym.set_rigid_body_color(
                            self.envs[i],
                            self._punyo_actor_id,
                            self.punyo_body_handles[side + "_floatie_" + str(j)],
                            gymapi.MESH_VISUAL,
                            # RGB color.
                            gymapi.Vec3(
                                self.states["floatie_pressure"][i][j+side_idx*TOTAL_NUM_FLOATIES_PER_ARM],
                                1 - self.states["floatie_pressure"][i][j+side_idx*TOTAL_NUM_FLOATIES_PER_ARM],
                                0.1,
                            ),
                        )
                    # Floatie 1.0
                    for part_name in ["a", "b"]:
                        self.gym.set_rigid_body_color(
                            self.envs[i],
                            self._punyo_actor_id,
                            self.punyo_body_handles[side + "_floatie_6" + part_name],
                            gymapi.MESH_VISUAL,
                            # RGB color.
                            gymapi.Vec3(
                                self.states["floatie_pressure"][i][6+side_idx*TOTAL_NUM_FLOATIES_PER_ARM],
                                1 - self.states["floatie_pressure"][i][6+side_idx*TOTAL_NUM_FLOATIES_PER_ARM],
                                0.1,
                            ),
                        )

                for j, side in enumerate(["l", "r"]):
                    self.gym.set_rigid_body_color(
                        self.envs[i],
                        self._punyo_actor_id,
                        self.punyo_body_handles[side + "_paw_bubble"],
                        gymapi.MESH_VISUAL,
                        # RGB color.
                        gymapi.Vec3(
                            self.states["paw_pressure"][i][j],
                            1 - self.states["paw_pressure"][i][j],
                            0.1,
                        ),
                    )


#####################################################################
###=========================jit functions=========================###
#####################################################################
# The PyTorch JIT compiler allows you to transform PyTorch code into an intermediate representation called TorchScript, which can then be further optimized and executed efficiently.
@torch.jit.script
def compute_task_reward(
    obs_buf, reset_buf, progress_buf, actions, previous_actions,
    states, max_episode_length, cumulative_successes
):
    # type: (Tensor, Tensor, Tensor, Tensor,Tensor, Dict[str, Tensor], float, Tensor) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor]
    # Compute per-env physical parameters.
    target_quat = states["target_quat"]
    target_pos = states["target_pos"]
    workspace_upper_limit_x = states["workspace_upper_limit_x"]
    workspace_lower_limit_x = states["workspace_lower_limit_x"]
    workspace_upper_limit_y = states["workspace_upper_limit_y"]
    workspace_lower_limit_y = states["workspace_lower_limit_y"]
    goal_xyz_tolerance = states["goal_xyz_tolerance"]
    goal_rot_tolerance = states["goal_rot_tolerance"]
    success_vel_tolerance = states["success_vel"]
    success_vel_tolerance_rot = states["success_vel_rot"]

    torque_threshold = states["torque_threshold"]
    contact_force_threshold = states["contact_force_threshold"]
    success_early_termination = states["success_early_termination"]
    max_cumulative_successes = states["max_cumulative_successes"]

    keypoint_reward_w = states["keypoint_reward_w"]
    translation_reward_w = states["translation_reward_w"]
    rotation_reward_w = states["rotation_reward_w"]
    action_penalty_w = states["action_penalty_w"]
    box_lin_vel_penalty_w = states["box_lin_vel_penalty_w"]
    box_rot_vel_penalty_w = states["box_rot_vel_penalty_w"]
    torque_penalty_w = states["torque_penalty_w"]
    action_diff_penalty_w = states["action_diff_penalty_w"]
    punyo_contact_penalty_w = states["punyo_contact_penalty_w"]
    box_contact_penalty_w = states["box_contact_penalty_w"]
    success_bonus_w = states["success_bonus_w"]

    # distance reward
    goal_dist = torch.norm(states["box_pos"] - target_pos, p=2, dim=-1)
    dist_rew = (1.0 / (torch.abs(goal_dist) + 0.1)) * translation_reward_w

    # keypoint reward
    keypoints = states["keypoints"]
    rotation_matrices = xyzw_quaternion_to_matrix(states["box_quat"])
    # keypoints in world frame.
    keypoints_transformed = torch.matmul(keypoints.unsqueeze(0), rotation_matrices.transpose(1, 2)) + states["box_pos"].unsqueeze(1)
    # target keypoints in world frame.
    target_rotation_matrices = xyzw_quaternion_to_matrix(target_quat)
    target_keypoints = torch.matmul(keypoints.unsqueeze(0), target_rotation_matrices.transpose(1, 2)) + target_pos.unsqueeze(1)
    # keypoint distances
    distances = torch.norm(target_keypoints - keypoints_transformed, dim=2)
    keypoint_sum_distances = distances.sum(dim=1)
    keypoint_rew = (1.0 / (torch.abs(keypoint_sum_distances) + 0.1)) * keypoint_reward_w

    # rotation reward
    # (TODO)jose-barreiros: Deprecate rotation reward in favor of keypoints.
    quat_diff = quat_mul(states["box_quat"], quat_conjugate(target_quat))  #xyzw
    rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )  #  Difference in radians between quaternions.
    rot_rew = (1.0 / (torch.abs(rot_dist) + 0.1)) * rotation_reward_w

    # Penalties for the robot actions, box velocity, and joint torques.
    action_penalty = torch.sum(actions**2, dim=-1) * (-action_penalty_w)
    action_diff_penalty = torch.sum((actions - previous_actions) ** 2, dim=-1) * (-action_diff_penalty_w)
    box_lin_vel_penalty = torch.sum(states["box_vel"][:,:3] ** 2, dim=-1) * (-box_lin_vel_penalty_w)
    box_rot_vel_penalty = torch.sum(states["box_vel"][:,3:] ** 2, dim=-1) * (-box_rot_vel_penalty_w)
    torque_penalty = torch.sum(states["dof_force"] ** 2, dim=-1) * (-torque_penalty_w)

    # Penalty for the contact forces.
    punyo_contact_mag = torch.norm(states["punyo_contact_forces"], p=2, dim=2)
    punyo_contact_penalty = torch.sum(punyo_contact_mag ** 2, dim=-1) * (-punyo_contact_penalty_w)
    box_contact_penalty = torch.sum(states["box_contact_forces"] ** 2, dim=-1) * (-box_contact_penalty_w)

    TABLE_HEIGHT = 0   # tabletop height in world frame.
    box_height = states["box_pos"][:, 2] - TABLE_HEIGHT
    box_dropped = box_height < 0.0
    drop_penalty = box_dropped * (-1.0)

    box_out_of_bound = torch.logical_or(
        torch.logical_or(
            torch.logical_or(
                (states["box_pos"][:, 0] > workspace_upper_limit_x),
                (states["box_pos"][:, 0] < workspace_lower_limit_x),
            ),
            (states["box_pos"][:, 1] > workspace_upper_limit_y),
        ),
        (states["box_pos"][:, 1] < workspace_lower_limit_y),
    )

    out_of_bound_penalty = box_out_of_bound * (-1.0)

    # Calculate success conditions per step.
    box_lin_vel_norm = torch.norm(states["box_vel"][:,:3], p=2, dim=-1)
    box_rot_vel_norm = torch.norm(states["box_vel"][:,3:], p=2, dim=-1)
    success_dist = goal_dist <= goal_xyz_tolerance
    success_rot = torch.abs(rot_dist) <= goal_rot_tolerance
    success_lin_vel = box_lin_vel_norm <= success_vel_tolerance
    success_rot_vel = box_rot_vel_norm <= success_vel_tolerance_rot
    step_success_condition = success_dist & success_rot & success_lin_vel & success_rot_vel

    # Reward bonus for success.
    success_bonus = step_success_condition * success_bonus_w

    # Number of cumulative successes after a reset.
    cumulative_successes += step_success_condition * 1

    rewards = (
        dist_rew
        + rot_rew
        + keypoint_rew
        + action_penalty
        + action_diff_penalty
        + box_lin_vel_penalty
        + box_rot_vel_penalty
        + torque_penalty
        + punyo_contact_penalty
        + box_contact_penalty
        + drop_penalty
        + out_of_bound_penalty
        + success_bonus
    )

    # Output individual reward components.
    # TODO(jose-barreiros): convert this to dictionary structure
    # that is compatible with JIT.
    individual_rewards = (
        dist_rew,
        rot_rew,
        keypoint_rew,
        action_penalty,
        action_diff_penalty,
        box_lin_vel_penalty,
        box_rot_vel_penalty,
        torque_penalty,
        punyo_contact_penalty,
        box_contact_penalty,
        drop_penalty,
        out_of_bound_penalty,
        success_bonus,
    )

    success_metrics = (
        goal_dist,
        rot_dist,
        box_lin_vel_norm,
        box_rot_vel_norm
    )

    # Compute resets:
    # Terminate if success.
    if success_early_termination:
        # TODO (jose-barreiros) Check for consecutive successes instead of cumulatives.
        # It is a better metric.
        reset_buf = reset_buf | (cumulative_successes >= max_cumulative_successes)

    # Terminate if the maximum episode length is reached.
    reset_buf = reset_buf | (progress_buf >= max_episode_length - 1)

    # Terminate if the box is dropped.
    reset_buf = reset_buf | (box_height < 0)

    # Terminate if the box is out of reach.
    reset_buf = reset_buf | (states["box_pos"][:, 0] > workspace_upper_limit_x)
    reset_buf = reset_buf | (states["box_pos"][:, 0] < workspace_lower_limit_x)
    reset_buf = reset_buf | (states["box_pos"][:, 1] > workspace_upper_limit_y)
    reset_buf = reset_buf | (states["box_pos"][:, 1] < workspace_lower_limit_y)

    # Terminate if a torque is above the threshold.
    reset_buf = reset_buf | torch.any(torch.abs(states['dof_force'][:, :]) > torque_threshold, dim=1)

    # Terminate if a robot contact violates the force threshold.
    reset_buf = reset_buf | torch.any(punyo_contact_mag > contact_force_threshold, dim=1)

    # Compute the sucess rate.
    # Note: Success is defined as the manipuland reaching
    # the goal pose in the last step of the episode within
    # a user-defined position, rotation, and velocity tolerances.
    is_last_episode = reset_buf.to(torch.bool)
    successes = (
        success_dist &
        success_rot &
        success_lin_vel &
        success_rot_vel &
        is_last_episode) * 1  # convert binary to float.
    return rewards, reset_buf, individual_rewards, successes, success_metrics, cumulative_successes
