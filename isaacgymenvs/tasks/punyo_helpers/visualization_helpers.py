"""
This script defines methods for Punyo-specific visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from datetime import datetime
from matplotlib import cm
import pandas as pd

class TrajectoryPlotter:
    """
    The visualizer object for trajectories.
    """

    def __init__(
        self,
        dt: float,
        max_episode_length: int,
        qpos_lb: np.ndarray,
        qpos_ub: np.ndarray,
        prefix="",
        save_data=False
    ):
        # Set the number of arms and the degrees of freedom per arm.
        self.NUM_ARMS = 2
        self.DOF_PER_ARM = 7
        self.ROBOT_DOF = self.NUM_ARMS * self.DOF_PER_ARM
        # 6 floatie 2.0s, 2 floatie 1.0 parts, and 1 paw.
        self.NUM_CONTACT_BODIES_PER_ARM = 6 + 2 + 1
        self.NUM_CONTACT_BODIES = self.NUM_ARMS * self.NUM_CONTACT_BODIES_PER_ARM
        # Initialize the trajectories.
        self.qpos = np.empty((0, self.ROBOT_DOF))
        self.qvel = np.empty((0, self.ROBOT_DOF))
        self.qpos_cmd = np.empty((0, self.ROBOT_DOF))
        self.qvel_cmd = np.empty((0, self.ROBOT_DOF))
        self.tau = np.empty((0, self.ROBOT_DOF))
        self.qerrors = np.empty((0, self.ROBOT_DOF))
        self.qvel_max = np.empty((0, self.ROBOT_DOF))
        self.fc = np.empty((0, self.NUM_CONTACT_BODIES))
        # Create figures for joint positions, velocities, torques, and max. speeds.
        self.fig1, self.axs1 = plt.subplots(
            self.DOF_PER_ARM, self.NUM_ARMS, figsize=(10, 10), constrained_layout=True
        )
        self.fig2, self.axs2 = plt.subplots(
            self.DOF_PER_ARM, self.NUM_ARMS, figsize=(10, 10), constrained_layout=True
        )
        self.fig3, self.axs3 = plt.subplots(
            self.DOF_PER_ARM, self.NUM_ARMS, figsize=(10, 10), constrained_layout=True
        )
        self.fig4, self.axs4 = plt.subplots(
            self.DOF_PER_ARM, self.NUM_ARMS, figsize=(10, 10), constrained_layout=True
        )
        self.fig5, self.axs5 = plt.subplots(
            self.DOF_PER_ARM, self.NUM_ARMS, figsize=(10, 10), constrained_layout=True
        )
        self.fig6, self.axs6 = plt.subplots(
            self.NUM_ARMS, figsize=(8, 6), constrained_layout=True
        )
        # Create a figure for contact force magnitudes.
        self.fig7, self.axs7 = plt.subplots(
            self.NUM_CONTACT_BODIES_PER_ARM, self.NUM_ARMS,
            figsize=(10, 12), constrained_layout=True
        )
        # Create colormap for the speed figure.
        self.colormap = cm.get_cmap("jet", self.DOF_PER_ARM)
        # Get env params.
        self.dt = dt
        self.max_episode_length = max_episode_length
        self.qpos_lb = qpos_lb
        self.qpos_ub = qpos_ub
        self.prefix = prefix
        self.save_data = save_data
        # Initialize the episode step counter.
        self.plot_ep_i = 0

    def plot(
        self,
        qpos_i: np.ndarray,
        qvel_i: np.ndarray,
        tau_i: np.ndarray,
        qpos_cmd_i: np.ndarray,
        qvel_max_i: np.ndarray,
        fc_i: np.ndarray,
    ):
        """
        This function accumulates trajectories until reaching the end of
        the episode. Once an episode is completed, the joint position, velocity,
        torque, and maximum speed trajectories as well as the contact force
        magnitudes exerted by the floaties and paws are plotted in different
        figures.
        """
        # Append the current values to the trajectories.
        self.qpos = np.vstack((self.qpos, qpos_i))
        self.qvel = np.vstack((self.qvel, qvel_i))
        self.qpos_cmd = np.vstack((self.qpos_cmd, qpos_cmd_i))
        self.tau = np.vstack((self.tau, tau_i))
        self.qvel_max = np.vstack((self.qvel_max, qvel_max_i))
        self.fc = np.vstack((self.fc, fc_i))
        # Calculate position error.
        self.qerrors = self.qpos_cmd - self.qpos
        # Estimate commanded velocity.
        self.qvel_cmd = self.qerrors/self.dt

        # Count the episode step.
        self.plot_ep_i += 1

        # Check for episode completion.
        if self.plot_ep_i >= self.max_episode_length:

            def calc_metrics(ref, act):
                errors = []
                max_abs_error = 0
                for i, el in enumerate(act):
                    error = ref[i] - el
                    if abs(error) > max_abs_error:
                        max_abs_error = abs(error)
                    errors.append(error)
                avg_abs_error = np.sum(np.abs(errors)) / len(errors)
                return avg_abs_error, max_abs_error

            def calc_mean_min_max(array):
                return np.mean(array), np.min(array), np.max(array)

            # Get the time list.
            time_arr = np.array(range(self.plot_ep_i)) * self.dt
            # Plot the trajectories.
            for arm_id in range(self.NUM_ARMS):
                for joint_id_in_arm in range(self.DOF_PER_ARM):
                    joint_id = arm_id * self.DOF_PER_ARM + joint_id_in_arm
                    # Calculate the performance metrics.
                    qpos_avg_abs_err, qpos_max_abs_err = calc_metrics(
                        ref=self.qpos_cmd[:, joint_id], act=self.qpos[:, joint_id]
                    )
                    v_mean, v_min, v_max = calc_mean_min_max(
                        self.qvel[:, joint_id])
                    vc_mean, vc_min, vc_max = calc_mean_min_max(
                        self.qvel_cmd[:, joint_id])
                    tau_mean, tau_min, tau_max = calc_mean_min_max(
                        self.tau[:, joint_id])
                    error_mean, error_min, error_max = calc_mean_min_max(
                        self.qerrors[:, joint_id])

                    # Plot the reference and actual positions per joint.
                    self.axs1[joint_id_in_arm, arm_id].plot(
                        time_arr, self.qpos[:, joint_id], "b-", label="act."
                    )
                    self.axs1[joint_id_in_arm, arm_id].plot(
                        time_arr, self.qpos_cmd[:, joint_id], "r-.", label="ref."
                    )
                    # Plot the joint velocities.
                    self.axs2[joint_id_in_arm, arm_id].plot(
                        time_arr, self.qvel[:, joint_id], "b-"
                    )
                    # Plot the joint torques.
                    self.axs3[joint_id_in_arm, arm_id].plot(
                        time_arr, self.tau[:, joint_id], "b-"
                    )
                    # Plot the commanded velocities.
                    self.axs4[joint_id_in_arm, arm_id].plot(
                        time_arr, self.qvel_cmd[:, joint_id], 'b-'
                        )
                    # Plot the joint error.
                    self.axs5[joint_id_in_arm, arm_id].plot(
                        time_arr, self.qerrors[:, joint_id], 'b-'
                    )
                    # Plot the max joint speeds for each arm.
                    self.axs6[arm_id].plot(
                        time_arr,
                        self.qvel_max[:, joint_id],
                        "-",
                        color=self.colormap(joint_id_in_arm),
                        label=f"J{joint_id}",
                    )
                    # Set the axis limits.
                    self.axs1[joint_id_in_arm, arm_id].set_xlim([0, time_arr[-1]])
                    self.axs2[joint_id_in_arm, arm_id].set_xlim([0, time_arr[-1]])
                    self.axs3[joint_id_in_arm, arm_id].set_xlim([0, time_arr[-1]])
                    self.axs4[joint_id_in_arm, arm_id].set_xlim([0, time_arr[-1]])
                    self.axs5[joint_id_in_arm, arm_id].set_xlim([0, time_arr[-1]])
                    self.axs6[arm_id].set_xlim([0, time_arr[-1]])

                    self.axs1[joint_id_in_arm, arm_id].set_ylim(
                        [self.qpos_lb[joint_id], self.qpos_ub[joint_id]]
                    )
                    self.axs2[joint_id_in_arm, arm_id].set_ylim(
                        [np.min(self.qvel) - 1, np.max(self.qvel) + 1]
                    )
                    self.axs3[joint_id_in_arm, arm_id].set_ylim(
                        [np.min(self.tau) - 1, np.max(self.tau) + 1]
                    )
                    self.axs4[joint_id_in_arm, arm_id].set_ylim(
                        [np.min(self.qvel_cmd) - 1, np.max(self.qvel_cmd) + 1]
                    )
                    self.axs5[joint_id_in_arm, arm_id].set_ylim(
                        [np.min(self.qerrors) - 0.1, np.max(self.qerrors) + 0.1]
                    )
                    # Set the position subplot titles.
                    self.axs1[joint_id_in_arm, arm_id].set_title(
                        f"avg. abs. err.={qpos_avg_abs_err:.3f}, "
                        f"max. abs. err.={qpos_max_abs_err:.3f}"
                    )
                    subplot_title = (f"mean: {v_mean:.3f}, "
                                     f"max: {v_max:.3f}, min: {v_min:.3f}")
                    self.axs2[joint_id_in_arm, arm_id].set_title(subplot_title)
                    subplot_title = (f"mean: {tau_mean:.3f}, "
                                     f"max: {tau_max:.3f}, min: {tau_min:.3f}")
                    self.axs3[joint_id_in_arm, arm_id].set_title(subplot_title)
                    subplot_title = (f"mean: {vc_mean:.3f}, "
                                     f"max: {vc_max:.3f}, min: {vc_min:.3f}")
                    self.axs4[joint_id_in_arm, arm_id].set_title(subplot_title)
                    subplot_title = (f"mean: {error_mean:.3f}, "
                                     f"max: {error_max:.3f}, min: {error_min:.3f}")
                    self.axs5[joint_id_in_arm, arm_id].set_title(subplot_title)

                    # Add label and grid for each axis.
                    self.axs1[joint_id_in_arm, arm_id].set_ylabel(f"J{joint_id}")
                    self.axs2[joint_id_in_arm, arm_id].set_ylabel(f"J{joint_id}")
                    self.axs3[joint_id_in_arm, arm_id].set_ylabel(f"J{joint_id}")
                    self.axs4[joint_id_in_arm, arm_id].set_ylabel(f"J{joint_id}")
                    self.axs5[joint_id_in_arm, arm_id].set_ylabel(f"J{joint_id}")
                    self.axs1[joint_id_in_arm, arm_id].grid(":")
                    self.axs2[joint_id_in_arm, arm_id].grid(":")
                    self.axs3[joint_id_in_arm, arm_id].grid(":")
                    self.axs4[joint_id_in_arm, arm_id].grid(":")
                    self.axs5[joint_id_in_arm, arm_id].grid(":")
                    # Remove the x tick labels except for the bottom plot.
                    if joint_id_in_arm < self.DOF_PER_ARM - 1:
                        plt.setp(
                            self.axs1[joint_id_in_arm, arm_id].get_xticklabels(),
                            visible=False,
                        )
                        plt.setp(
                            self.axs2[joint_id_in_arm, arm_id].get_xticklabels(),
                            visible=False,
                        )
                        plt.setp(
                            self.axs3[joint_id_in_arm, arm_id].get_xticklabels(),
                            visible=False,
                        )
                        plt.setp(
                            self.axs4[joint_id_in_arm, arm_id].get_xticklabels(),
                            visible=False,
                        )
                        plt.setp(
                            self.axs5[joint_id_in_arm, arm_id].get_xticklabels(),
                            visible=False,
                        )
                    else:
                        self.axs1[joint_id_in_arm, arm_id].set_xlabel("Time [s]")
                        self.axs2[joint_id_in_arm, arm_id].set_xlabel("Time [s]")
                        self.axs3[joint_id_in_arm, arm_id].set_xlabel("Time [s]")
                        self.axs4[joint_id_in_arm, arm_id].set_xlabel("Time [s]")
                        self.axs5[joint_id_in_arm, arm_id].set_xlabel("Time [s]")
                # Plot the contact force magnitudes for the robot bodies.
                for arm_id, arm_name in enumerate(["l", "r"]):
                    # Plot the contact force for each floatie.
                    for floatie_id in range(self.NUM_CONTACT_BODIES_PER_ARM - 1):
                        body_idx = arm_id * self.NUM_CONTACT_BODIES_PER_ARM + floatie_id
                        if floatie_id == 6:
                            floatie_name = arm_name + '_floatie_6a'
                        elif floatie_id == 7:
                            floatie_name = arm_name + '_floatie_6b'
                        else:
                            floatie_name = arm_name + '_floatie_' + str(floatie_id)
                        self.axs7[floatie_id, arm_id].plot(
                            time_arr, self.fc[:, body_idx], 'b-'
                        )
                        # Make the plot pretty.
                        self.axs7[floatie_id, arm_id].set_xlim([0, time_arr[-1]])
                        plt.setp(
                            self.axs7[floatie_id, arm_id].get_xticklabels(),
                            visible=False,
                        )
                        self.axs7[floatie_id, arm_id].set_ylim(
                            [np.min(self.fc) - 1, np.max(self.fc) + 1]
                        )
                        self.axs7[floatie_id, arm_id].set_ylabel(floatie_name)
                        self.axs7[floatie_id, arm_id].grid(":")
                    # Plot the contact force for the paw.
                    paw_name = arm_name + '_paw'
                    plot_id=floatie_id+1
                    self.axs7[plot_id, arm_id].plot(
                        time_arr, self.fc[:, body_idx+1], 'b-'
                    )
                    # Make the plot pretty.
                    self.axs7[plot_id, arm_id].set_xlim([0, time_arr[-1]])
                    self.axs7[plot_id, arm_id].set_ylim(
                        [np.min(self.fc) - 5, np.max(self.fc) + 5]
                    )
                    self.axs7[plot_id, arm_id].set_xlabel("Time [s]")
                    self.axs7[plot_id, arm_id].set_ylabel(paw_name)
                    self.axs7[plot_id, arm_id].grid(":")
            # Add title, legend, labels, and grid.
            self.fig1.suptitle("Isaac - joint positions [rad]")
            self.fig2.suptitle("Isaac - joint velocities [rad/s]")
            self.fig3.suptitle("Isaac - joint torques [N-m]")
            self.fig4.suptitle("Isaac - commanded velocities [rad/s]")
            self.fig5.suptitle("Isaac - joint position errors [rad]")
            self.fig6.suptitle("Isaac - max. joint speeds across environments [rad/s]")
            self.fig7.suptitle("Isaac - contact force magnitudes [N]")
            ax_handles, ax_labels = self.axs1[0, 0].get_legend_handles_labels()
            self.fig1.legend(ax_handles, ax_labels, loc="right")
            ax_handles, ax_labels = self.axs6[0].get_legend_handles_labels()
            self.fig6.legend(ax_handles, ax_labels, loc="right")
            self.axs6[0].set_ylabel("Left arm")
            self.axs6[1].set_ylabel("Right arm")
            plt.setp(self.axs6[0].get_xticklabels(), visible=False)
            self.axs6[1].set_xlabel("Time [s]")
            self.axs6[0].grid(":")
            self.axs6[1].grid(":")
            # Set the log path.
            date = datetime.now().strftime("%Y%m%d_%H_%M")
            data_log_path = os.environ["HOME"] + f"/punyo_CI/isaac_test_{date}/"
            if not os.path.isdir(data_log_path):
                os.makedirs(data_log_path, exist_ok=True)
            # Save the figures.
            self.fig1.savefig(data_log_path + f"qpos.png")
            self.fig2.savefig(data_log_path + f"qvel.png")
            self.fig3.savefig(data_log_path + f"tau.png")
            self.fig5.savefig(data_log_path + f"qerror.png")
            self.fig6.savefig(data_log_path + f"max_qvel.png")
            self.fig7.savefig(data_log_path + f"fc.png")
            print(f"Saving the figures at {data_log_path}")
            if self.save_data:
                data = ([time_arr] +
                        [self.qpos_cmd] +
                        [self.qvel_cmd] +
                        [time_arr] +
                        [self.qpos] +
                        [self.qerrors] +
                        [self.qvel] +
                        [self.tau] +
                        [self.fc])
                data_transpose = [list(i) for i in zip(*data)]
                columns = [
                    "action_times", "actions", "commanded_velocities",
                    "observation_times", "positions",
                    "position_errors", "velocities", "torques",
                    "contact_forces"]
                df = pd.DataFrame(data_transpose, columns=columns)
                df.to_pickle(data_log_path + 'data.pkl')
                df.to_csv(data_log_path + 'data.csv', index=False)

            # Draw and prompt user for exit.
            plt.draw()
            plt.pause(1e-3)
            input("A full episode has been completed. Hit enter to exit.")
            sys.exit()
