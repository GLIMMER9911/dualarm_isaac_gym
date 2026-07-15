import numpy as np
import os
import glob
#import pickle5 as pickle
import pickle
import pdb
import torch

from isaacgym.torch_utils import (
    to_torch
)
from isaacgymenvs.utils.torch_jit_utils import (
    quaternion_to_positive_w,
    slerp,
    quat_from_euler_xyz,
)

import os

class Robot_Motion:
    def __init__(self) -> None:
        self.tensor = None
        self.dof_vels = None

class MazeMotionLib:
    def __init__(
            self, motion_file_path,
            num_dofs, device,
            default_dt,
            ignore_before_idx=0,
            ) -> None:
        self._num_dof = num_dofs
        self._device = device
        self.ignore_before_idx = ignore_before_idx
        self.default_dt = default_dt

        self._load_motions(motion_file_path)
        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)
        self.unique_ids = np.unique(self.motion_ids.to("cpu").numpy())

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        m = self.num_motions()
        motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)

        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)

        dof_pos0 = torch.empty([n, self._num_dof], dtype=float, device=self._device)
        dof_pos1 = torch.empty([n, self._num_dof], dtype=float, device=self._device)
        box_pos0 = torch.empty([n, 3], dtype=float, device=self._device)
        box_pos1 = torch.empty([n, 3], dtype=float, device=self._device)
        box_rot0 = torch.empty([n, 4], dtype=float, device=self._device)
        box_rot1 = torch.empty([n, 4], dtype=float, device=self._device)
        dof_vel = torch.empty([n, self._num_dof], dtype=float, device=self._device)
        ball_pressures = torch.empty([n, 1], dtype=float, device=self._device)
        robot_binary_contact = torch.empty([n, 5], dtype=float, device=self._device)
        box_binary_contact = torch.empty([n, 5], dtype=float, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        for uid in self.unique_ids:
            ids = np.where(motion_ids == uid)
            curr_motion = self._motions[uid]

            dof_pos0[ids, :] = curr_motion.dof_pos[frame_idx0[ids]]
            dof_pos1[ids, :] = curr_motion.dof_pos[frame_idx1[ids]]

            box_pos0[ids, :] = curr_motion.box_pos[frame_idx0[ids]]
            box_pos1[ids, :] = curr_motion.box_pos[frame_idx1[ids]]

            box_rot0[ids, :] = curr_motion.box_rot[frame_idx0[ids]]
            box_rot1[ids, :] = curr_motion.box_rot[frame_idx1[ids]]

            if curr_motion.dof_vels is not None:
                dof_vel[ids, :] = curr_motion.dof_vels[frame_idx0[ids]]

            if curr_motion.ball_pressures is not None:
                ball_pressures[ids, :] = curr_motion.ball_pressures[frame_idx0[ids]]

            if curr_motion.robot_binary_contact is not None:
                robot_binary_contact[ids, :] = curr_motion.robot_binary_contact[frame_idx0[ids]]
            if curr_motion.box_binary_contact is not None:
                box_binary_contact[ids, :] = curr_motion.box_binary_contact[frame_idx0[ids]]

        blend = to_torch(np.expand_dims(blend, axis=-1))

        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        box_pos = (1.0 - blend) * box_pos0 + blend * box_pos1

        box_rot = quaternion_to_positive_w(
            slerp(box_rot0, box_rot1, blend))

        # Check for existence of observation types.
        if curr_motion.dof_vels is None:
            dof_vel = None
        if curr_motion.robot_binary_contact is None:
            robot_binary_contact = None
        if curr_motion.box_binary_contact is None:
            box_binary_contact = None
        if curr_motion.ball_pressures is not None:
            # Threshold the force magnitude to get binary contact state.
            ball_pressures[ball_pressures>0] = 1
        else:
            ball_pressures = None

        return box_pos, box_rot, dof_pos, dof_vel, robot_binary_contact, box_binary_contact

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)

        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            with open(curr_file, 'rb') as file:
                curr_motion_pd = pickle.load(file)

            curr_motion = Robot_Motion()
            curr_motion.tensor = torch.from_numpy(np.stack(curr_motion_pd['observations']['state'].values)[self.ignore_before_idx:,:]).to(self._device)
            curr_motion.num_joints = 3
            curr_motion.dof_pos = curr_motion.tensor[:,:curr_motion.num_joints]
            if curr_motion.tensor.shape[1] > curr_motion.num_joints:
                curr_motion.dof_vels = curr_motion.tensor[:,curr_motion.num_joints:]
            else:
                curr_motion.dof_vels = None

            manipuland_rpyxyz = torch.from_numpy(np.stack(curr_motion_pd['observations']['manipuland_rpyxyz'].values)[self.ignore_before_idx:,:]).to(self._device)
            roll, pitch, yaw = manipuland_rpyxyz[:,0:1], manipuland_rpyxyz[:,1:2], manipuland_rpyxyz[:,2:3]
            curr_motion.box_pos = manipuland_rpyxyz[:,3:6]
            curr_motion.box_rot = quat_from_euler_xyz(roll, pitch, yaw)

            try:
                curr_motion.dt = curr_motion_pd[("config","config")][0]["gym_timestep"]
            except:
                # Uses the default dt if the demosntration file does not have
                # the key [("config","config")]["gym_timestep"]
                curr_motion.dt = self.default_dt
                print(f"Motion file does NOT have gym_timestep. Using default motion dt: {curr_motion.dt}")
            print(f"Motion time step (dt) set to {curr_motion.dt}")

            curr_motion.fps = 1.0 / curr_motion.dt

            try:
                ball_pressures = torch.from_numpy(np.stack(curr_motion_pd['observations']['ball_pressures'].values)[self.ignore_before_idx:,:]).to(self._device)
                curr_motion.ball_pressures = ball_pressures
            except:
                curr_motion.ball_pressures = None
                print(f"Motion file does NOT have ball_pressures")

            # Extract the binary contact states if available.
            try:
                robot_binary_contact = torch.from_numpy(np.stack(curr_motion_pd['observations']['robot_binary_contact'].values)[self.ignore_before_idx:,:]).to(self._device)
                curr_motion.robot_binary_contact = robot_binary_contact
            except:
                curr_motion.robot_binary_contact = None
                print(f"Motion file does NOT have robot_binary_contact")
            try:
                box_binary_contact = torch.from_numpy(np.stack(curr_motion_pd['observations']['box_binary_contact'].values)[self.ignore_before_idx:,:]).to(self._device)
                curr_motion.box_binary_contact = box_binary_contact
            except:
                curr_motion.box_binary_contact = None
                print(f"Motion file does NOT have box_binary_contact")

            motion_fps = curr_motion.fps
            curr_dt = curr_motion.dt

            num_frames = curr_motion.tensor.shape[0]
            curr_len = curr_dt * (num_frames - 1)  # in sec.

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        # Normalize the weights.
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        avg_dt = np.sum(self._motion_dt) / num_motions
        print(f"Loaded {num_motions} motions with a total length of {total_len}sec."
              f"average lenght {total_len/num_motions} sec. and average frames {total_len/(num_motions*avg_dt)}")

    def _fetch_motion_files(self, motion_file_path):

        # Get a list of all .pkl files in the folder and its subfolders
        motion_files = glob.glob(os.path.join(motion_file_path, '**', '*.pkl'), recursive=True)
        from icecream import ic
        ic(motion_files)
        motion_weights = [1.0]*len(motion_files)

        return motion_files, motion_weights

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        assert np.all(phase <= 1)
        assert np.all(phase >= 0)

        frame_idx0 = (phase * (num_frames - 1)).astype(int)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)
        # Calculates the weight (blend) of the frame_idx0.
        blend = (time - frame_idx0 * dt) / dt

        assert np.all(blend <= 1)
        assert np.all(blend >= 0)

        return frame_idx0, frame_idx1, blend
