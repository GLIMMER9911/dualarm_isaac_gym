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

class MotionLib:
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
        lhand_pos0 = torch.empty([n, 3], dtype=float, device=self._device)
        lhand_pos1 = torch.empty([n, 3], dtype=float, device=self._device)
        rhand_pos0 = torch.empty([n, 3], dtype=float, device=self._device)
        rhand_pos1 = torch.empty([n, 3], dtype=float, device=self._device)
        lhand_rot0 = torch.empty([n, 4], dtype=float, device=self._device)
        lhand_rot1 = torch.empty([n, 4], dtype=float, device=self._device)
        rhand_rot0 = torch.empty([n, 4], dtype=float, device=self._device)
        rhand_rot1 = torch.empty([n, 4], dtype=float, device=self._device)
        dof_vel = torch.empty([n, self._num_dof], dtype=float, device=self._device)
        paw_pressures = torch.empty([n, 2], dtype=float, device=self._device)
        floatie_pressures = torch.empty([n, 14], dtype=float, device=self._device)
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

            box_rot0[ids, :] = curr_motion.box_rot[frame_idx0[ids]].squeeze(1)
            box_rot1[ids, :] = curr_motion.box_rot[frame_idx1[ids]].squeeze(1)

            if curr_motion.lhand_pos is not None:
                lhand_pos0[ids, :] = curr_motion.lhand_pos[frame_idx0[ids]]
                lhand_pos1[ids, :] = curr_motion.lhand_pos[frame_idx1[ids]]

            if curr_motion.rhand_pos is not None:
                rhand_pos0[ids, :] = curr_motion.rhand_pos[frame_idx0[ids]]
                rhand_pos1[ids, :] = curr_motion.rhand_pos[frame_idx1[ids]]

            if curr_motion.lhand_rot is not None:
                lhand_rot0[ids, :] = curr_motion.lhand_rot[frame_idx0[ids]]
                lhand_rot1[ids, :] = curr_motion.lhand_rot[frame_idx1[ids]]

            if curr_motion.rhand_rot is not None:
                rhand_rot0[ids, :] = curr_motion.rhand_rot[frame_idx0[ids]]
                rhand_rot1[ids, :] = curr_motion.rhand_rot[frame_idx1[ids]]

            if curr_motion.dof_vels is not None:
                dof_vel[ids, :] = curr_motion.dof_vels[frame_idx0[ids]]

            if curr_motion.paw_pressures is not None:
                paw_pressures[ids, :] = curr_motion.paw_pressures[frame_idx0[ids]]

            if curr_motion.floatie_pressures is not None:
                floatie_pressures[ids, :] = curr_motion.floatie_pressures[frame_idx0[ids]]

        blend = to_torch(np.expand_dims(blend, axis=-1))

        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        box_pos = (1.0 - blend) * box_pos0 + blend * box_pos1

        if curr_motion.lhand_pos is not None:
            lhand_pos = (1.0 - blend) * lhand_pos0 + blend * lhand_pos1
        else:
            lhand_pos = None

        if curr_motion.rhand_pos is not None:
            rhand_pos = (1.0 - blend) * rhand_pos0 + blend * rhand_pos1
        else:
            rhand_pos = None

        if curr_motion.lhand_rot is not None:
            # Make sure the w component of a quaternion is always positive
            lhand_rot = quaternion_to_positive_w(
                slerp(lhand_rot0, lhand_rot1, blend))
        else:
            lhand_rot = None

        if curr_motion.rhand_rot is not None:
            rhand_rot = quaternion_to_positive_w(
                slerp(rhand_rot0, rhand_rot1, blend))
        else:
            rhand_rot = None

        box_rot = quaternion_to_positive_w(
            slerp(box_rot0, box_rot1, blend))

        if curr_motion.dof_vels is None:
            dof_vel = None

        if curr_motion.paw_pressures is not None:
            paw_pressures[paw_pressures>0] = 1
        else:
            paw_pressures = None

        if curr_motion.floatie_pressures is not None:
            floatie_pressures[floatie_pressures>0] = 1
        else:
            floatie_pressures = None

        return box_pos, box_rot, dof_pos, dof_vel, lhand_pos, lhand_rot, rhand_pos, rhand_rot, paw_pressures, floatie_pressures

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
            curr_motion.num_joints = self._num_dof
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
                end_effectors_rpyxyz = torch.from_numpy(np.stack(curr_motion_pd['observations']['end_effectors_rpyxyz'].values)[self.ignore_before_idx:,:]).to(self._device)
                lhand_state_ = end_effectors_rpyxyz[:,:6]
                rhand_state_ = end_effectors_rpyxyz[:,6:]
                lhand_state = torch.cat((lhand_state_[:,3:],quat_from_euler_xyz(lhand_state_[:,0], lhand_state_[:,1], lhand_state_[:,2]).squeeze(1)),axis=1)
                rhand_state = torch.cat((rhand_state_[:,3:],quat_from_euler_xyz(rhand_state_[:,0], rhand_state_[:,1], rhand_state_[:,2]).squeeze(1)),axis=1)
                curr_motion.lhand_pos = lhand_state[:,:3]
                curr_motion.lhand_rot = lhand_state[:,3:]
                curr_motion.rhand_pos = rhand_state[:,:3]
                curr_motion.rhand_rot = rhand_state[:,3:]
            except:
                curr_motion.lhand_pos = None
                curr_motion.lhand_rot = None
                curr_motion.rhand_pos = None
                curr_motion.rhand_rot = None

            try:
                paw_pressures = torch.from_numpy(np.stack(curr_motion_pd['observations']['paw_pressures'].values)[self.ignore_before_idx:,:]).to(self._device)
                curr_motion.paw_pressures = paw_pressures
            except:
                curr_motion.paw_pressures = None

            try:
                floatie_pressures = torch.from_numpy(np.stack(curr_motion_pd['observations']['arm_pressures'].values)[self.ignore_before_idx:,:]).to(self._device)
                curr_motion.floatie_pressures = floatie_pressures
            except:
                curr_motion.floatie_pressures = None

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
