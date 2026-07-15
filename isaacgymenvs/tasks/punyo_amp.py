import math
import pdb
from enum import Enum

import numpy as np
import torch
from gym import spaces
from isaacgym import gymtorch
# from tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.tasks.amp.punyo_amp_base import PunyoAMPBase

from isaacgym.torch_utils import (
    quat_mul,
)


class PunyoAMP(PunyoAMPBase):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
        action_mode="delta_q",
    ):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = PunyoAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2

        # Parameters for box's initial pose randomization
        self.x_range = self.cfg["env"]["object"]["xRange"]
        self.y_range = self.cfg["env"]["object"]["yRange"]
        self.yaw_range = self.cfg["env"]["object"]["yawRange"]
        assert self.yaw_range < 2 * np.pi, "yaw range is outside [0, 2*pi]"

        # Check that the initial positions are inside the workspace limits.
        task_type = self.cfg["task"]["task_type"]
        TOLERANCE = 0.001
        limit_x_points = [
            self.cfg["task"][task_type]["default_x"] + self.x_range + TOLERANCE,
            self.cfg["task"][task_type]["default_x"] - self.x_range - TOLERANCE]
        limit_y_points = [
            self.cfg["task"][task_type]["default_y"] + self.y_range + TOLERANCE,
            self.cfg["task"][task_type]["default_y"] - self.y_range - TOLERANCE]

        message = "Initial manipuland position is out of workspace range "
        for x in limit_x_points:
            assert x < self.cfg["task"][task_type]["workspace_upper_limit_x"], message + "x_upper"
            assert x > self.cfg["task"][task_type]["workspace_lower_limit_x"], message + "x_lower"
        for y in limit_y_points:
            assert y < self.cfg["task"][task_type]["workspace_upper_limit_y"], message + "y_upper"
            assert y > self.cfg["task"][task_type]["workspace_lower_limit_y"], message + "y_lower"

        self.reward_settings = {}

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
            action_mode=action_mode,
        )

        self._load_motion(self.motion_dir_path)

        # Get the size of the observation space based on the user-specified config.
        self._amp_obs = cfg["task"][task_type]["ampObservation"]
        self.amp_state_types = self._map_obs_types_to_state_types(self._amp_obs)
        num_obs = self._get_obs_size(self._amp_obs)
        self._num_amp_obs_per_step = num_obs

        self.num_amp_obs = self._num_amp_obs_steps * self._num_amp_obs_per_step

        self._amp_obs_space = spaces.Box(
            np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf
        )

        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None

    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

    def get_num_amp_obs(self):
        return self.num_amp_obs

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert self._amp_obs_demo_buf.shape[0] == num_samples

        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(
            np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps]
        )
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -self.dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = np.clip(motion_times + time_steps, 0, None)

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()

        (
            box_pos,
            box_quat,
            dof_pos,
            dof_vel,
            lhand_pos,
            lhand_quat,
            rhand_pos,
            rhand_quat,
            paw_pressure,
            floatie_pressure,
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times)

        # Set the observation types based on the user-specified config.
        _obs = self._construct_amp_obs(
            box_pos, box_quat,
            dof_pos, dof_vel,
            lhand_pos, lhand_quat,
            rhand_pos, rhand_quat,
            paw_pressure,
            floatie_pressure,
            )
        obs = torch.cat(_obs, axis=-1)

        assert (
            obs.shape[-1] == self._num_amp_obs_per_step
        ), "Number of observations does not match the number of observations per step."

        amp_obs_demo = build_amp_observations(obs)

        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float,
        )

    def _load_motion(self, motion_file_path):
        self._motion_lib = MotionLib(
            motion_file_path=motion_file_path,
            num_dofs=self.num_dofs,
            device=self.device,
            default_dt=self.cfg["env"]["asset"]["motion"]["default_dt"],
            ignore_before_idx=self.cfg["env"]["asset"]["motion"]["ignore_before_idx"],
        )

    def _randomize_manipuland_pose(self, env_ids, demo_idx):
        """
        Randomize the manipuland pose for a set of environments given
        by env_ids. demo_idx specifies a random demonstration index
        per environment for the default orientation.
        """
        self._box_state[env_ids, 0:1] += (
            self.x_range * 2.0 * (
                torch.rand(len(env_ids), 1, device=self.device) - 0.5
            )
        )
        self._box_state[env_ids, 1:2] += (
            self.y_range * 2.0 * (
                torch.rand(len(env_ids), 1, device=self.device) - 0.5
            )
        )
        # Sample a random rotation angle for the box (w.r.t. the z axis) and get
        # its corresponding quaternion Default quaternion
        default_quat = torch.cat(
            [self.box_default_poses[i][:, 3:7] for i in demo_idx], dim=0
        )

        angles = (
            2
            * (torch.rand(len(env_ids), device=self.device) - 0.5)
            * self.yaw_range
            * math.pi
        )
        sin_angles = torch.sin(angles / 2)
        cos_angles = torch.cos(angles / 2)

        self._box_state[env_ids, 3:7] = quat_mul(
            default_quat,
            torch.stack(
                (
                    torch.zeros_like(angles),
                    torch.zeros_like(angles),
                    sin_angles,
                    cos_angles,
                ),
                dim=1,
            ),
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)

    def _reset_actors(self, env_ids):
        # Default: Set the robot to the default state (if any; start of the demonstration otherwise),
        # and set the box to the default state (if any) plus specified disturbance.
        # Start: Set the robot and the box to the start state of the demonstration.
        # Random: Set the robot and the box to a random state of the demonstration.
        # Hybrid: A combination of Default and Random.
        if self._state_init == PunyoAMP.StateInit.Default:
            self._reset_default(env_ids)
        elif (
            self._state_init == PunyoAMP.StateInit.Start
            or self._state_init == PunyoAMP.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == PunyoAMP.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self.cumulative_successes[env_ids] = 0

    def _reset_default(self, env_ids):
        # Select a random demonstration index per environment.
        demo_idx = torch.randint(
            0, len(self.motion_files), (len(env_ids),), device=self.device
        )

        self._box_state[env_ids] = torch.cat(
            [self.box_default_poses[i] for i in demo_idx], dim=0
        )

        # Perturb the manipuland pose if domain randomization is enabled.
        if self.randomize:
            self._randomize_manipuland_pose(env_ids=env_ids, demo_idx=demo_idx)

        # Reset the initial joint pose to the demo start poses.
        self._dof_pos[env_ids, :] = torch.cat(
            [self.punyo_default_dof_poses[i] for i in demo_idx], dim=0
        )
        # Reset the initial joint velocities to zero.
        self._dof_vel[env_ids, :] = torch.zeros_like(self._dof_vel[env_ids])
        # Reset the initial joint position commands to the initial positions.
        self.punyo_dof_pos_targets[env_ids, : self.num_punyo_dofs] = torch.clone(
            self._dof_pos[env_ids, :]
        )

        # Set the robot state and commands for all environments being reset.
        punyo_dof_indices_int32 = self._global_indices[
            env_ids, self._punyo_actor_id].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(punyo_dof_indices_int32),
            len(punyo_dof_indices_int32),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.punyo_dof_pos_targets),
            gymtorch.unwrap_tensor(punyo_dof_indices_int32),
            len(punyo_dof_indices_int32),
        )

        # Set the actor root states for all environments being reset.
        actor_root_indices_int32 = self._global_indices[env_ids, :].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(actor_root_indices_int32),
            len(actor_root_indices_int32),
        )

        self._reset_default_env_ids = env_ids

    def _reset_ref_state_init(self, env_ids, default_ids=None):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (
            self._state_init == PunyoAMP.StateInit.Random
            or self._state_init == PunyoAMP.StateInit.Hybrid
        ):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == PunyoAMP.StateInit.Start:
            motion_times = np.zeros(num_envs)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )

        (
            box_pos,
            box_quat,
            dof_pos,
            dof_vel,
            lhand_pos,
            lhand_quat,
            rhand_pos,
            rhand_quat,
            paw_pressure,
            floatie_pressure,
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times)
        self._set_env_state(
            env_ids=env_ids,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            box_pos=box_pos,
            box_quat=box_quat,
            default_ids=default_ids,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(
            np.array([self._hybrid_init_prob] * num_envs), device=self.device
        )
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids, default_ids=default_reset_ids)

    def _set_env_state(self, env_ids, dof_pos, dof_vel, box_pos, box_quat, default_ids):
        """
        Resets the initial state of each environment.
        Parameters:
            env_ids: The IDs of the environments to be reset to Random state.
            default_ids: The IDs of the environments to be reset to the default state.
        """
        self._dof_pos[env_ids, :] = dof_pos
        self._dof_vel[env_ids, :] = torch.zeros_like(dof_pos, device=self.device)

        self._box_pos[env_ids, :] = box_pos
        self._box_quat[env_ids, :] = box_quat
        # Set zero velocity.
        self._box_state[env_ids, box_pos.shape[1] + box_quat.shape[1] :] = torch.zeros(
            (box_pos.shape[0], 6), device=self.device
        )

        self.punyo_dof_pos_targets[env_ids, : self.num_punyo_dofs] = dof_pos

        if default_ids is not None:
            # Select a random demonstration index per environment.
            demo_idx = torch.randint(
                0, len(self.motion_files), (len(default_ids),), device=self.device
            )

            self._dof_pos[default_ids, :] = torch.cat(
                [self.punyo_default_dof_poses[i] for i in demo_idx], dim=0
            )
            self._dof_vel[default_ids, :] = torch.zeros_like(self._dof_vel[default_ids])
            self.punyo_dof_pos_targets[default_ids, : self.num_punyo_dofs] = torch.clone(
                self._dof_pos[default_ids, :]
            )

            self._box_state[default_ids] = torch.cat(
                [self.box_default_poses[i] for i in demo_idx], dim=0
            )

            # Perturb the manipuland pose if domain randomization is enabled.
            if self.randomize:
                self._randomize_manipuland_pose(env_ids=default_ids, demo_idx=demo_idx)

            all_ids = torch.cat((env_ids, default_ids), dim=0)
        else:
            all_ids = env_ids

        multi_env_ids_punyo_int32 = self._global_indices[all_ids, 0].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_punyo_int32),
            len(multi_env_ids_punyo_int32),
        )

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.punyo_dof_pos_targets),
            gymtorch.unwrap_tensor(multi_env_ids_punyo_int32),
            len(multi_env_ids_punyo_int32),
        )

        multi_env_ids_int32 = self._global_indices[all_ids, :].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if len(self._reset_default_env_ids) > 0:
            self._init_amp_obs_default(self._reset_default_env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(
                self._reset_ref_env_ids,
                self._reset_ref_motion_ids,
                self._reset_ref_motion_times,
            )

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs

    def _construct_amp_obs(
            self,
            box_pos, box_quat,
            dof_pos, dof_vel,
            lhand_pos, lhand_quat,
            rhand_pos, rhand_quat,
            paw_pressure,
            floatie_pressure,
            ):
        """
        Packs the AMP observations in an array.
        """
        _obs = []
        for ob in self._amp_obs:
            if "robot_dof" == ob:
                _obs.append(dof_pos)
            elif "robot_vel" == ob:
                _obs.append(dof_vel)
            elif "box_pose" == ob:
                _obs += [box_pos, box_quat]
            elif "ee_pose" == ob:
                _obs += [lhand_pos, lhand_quat, rhand_pos, rhand_quat]
            elif "ee_binary_contact" == ob:
                _obs.append(paw_pressure)
            elif "floatie_binary_contact" == ob:
                _obs.append(floatie_pressure)
            else:
                Exception("Unknown observation type: {}".format(ob))
        return _obs

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = np.tile(
            np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1]
        )
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()

        (
            box_pos,
            box_quat,
            dof_pos,
            dof_vel,
            lhand_pos,
            lhand_quat,
            rhand_pos,
            rhand_quat,
            paw_pressure,
            floatie_pressure,
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times)

        _obs = self._construct_amp_obs(
            box_pos, box_quat,
            dof_pos, dof_vel,
            lhand_pos, lhand_quat,
            rhand_pos, rhand_quat,
            paw_pressure,
            floatie_pressure,
            )
        obs = torch.cat(_obs, axis=-1)
        amp_obs_demo = build_amp_observations(obs)

        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(
            self._hist_amp_obs_buf[env_ids].shape
        )

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]

    def _compute_amp_observations(self, env_ids=None):
        _obs = self._compute_obs_by_type(self.amp_state_types, env_ids)

        if env_ids is None:
            self._curr_amp_obs_buf[:] = build_amp_observations(_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(_obs)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def build_amp_observations(obs):
    # type: (Tensor) -> Tensor
    return obs
