import argparse
import isaacgymenvs
import isaacgym
import os
import pdb
import time
import wandb

import matplotlib.pyplot as plt
import numpy as np

import torch
import yaml
from isaacgym import gymapi

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.policy_helpers import RlGamesPolicy

from isaacgym.torch_utils import (
    tensor_clamp,
)

def main():
    np.set_printoptions(precision = 3)

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--amp_task_name',
                        help="Name of the AMP task", type=str,
                        default="PunyoV2AMP",
                        choices=["PunyoV2AMP"])
    parser.add_argument('--num_envs',
                        help="Number of environments to simulate",
                        type=int, default=9)
    parser.add_argument('--train_config_file',
                        help="path to the train config yaml file."
                        "If provided, this overrides the config file"
                        "obtained from the wandb handle.")
    parser.add_argument('--base_config_file',
                        help="path to the base config yaml file."
                        "If provided, this overrides the config file"
                        "obtained from the wandb handle.")
    parser.add_argument('--show_floatie_contact',
                        help="highlights the floaties when in contact.",
                        action='store_true')
    parser.add_argument('--plot_trajectories',
                        help="plots the joint trajectories and the contact force magnitudes of the floaties and the paws for the last environment.",
                        action='store_true')
    parser.add_argument("--gpu_id",
                        help="GPU device ID: typically 0 or 1.",
                        default='0', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--action_mode",
                        help="Action types",
                        type=str, default="delta_q", choices=["q", "delta_q"])
    parser.add_argument("--action_scale", type=float,
                        help="Action scale that overwrites the one in the config if set")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test_success_metrics", action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--checkpoint_file',
                    help="path to policy checkpoint file")
    group.add_argument("--wandb_handle", type=str,
                        help="wandb URL pointing to the wandb run. Format wandb url::model_id")
    args = parser.parse_args()

    # Process the wandb handle if provided.
    if args.wandb_handle:
        assert not args.test
        assert not args.checkpoint_file

        wandb_ = wandb.Api()
        url = args.wandb_handle.split("::")[0]
        model_id = args.wandb_handle.split("::")[1]
        url = url.split("://")[1]
        parsed_url = url.split("/")
        entity =parsed_url[1]
        project =parsed_url[2]
        run_id =parsed_url[4]

        artifact_path = f"{entity}/{project}/{model_id}_checkpoint_{run_id}:latest"
        checkpoint_artifact = wandb_.artifact(artifact_path)

        artifact_path = f"{entity}/{project}/demonstrations_{run_id}:latest"
        demonstrations_artifact = wandb_.artifact(artifact_path)

        artifact_path = f"{entity}/{project}/configs_{run_id}:latest"
        configs_artifact = wandb_.artifact(artifact_path)

        demo_path = demonstrations_artifact.download()
        config_path = configs_artifact.download()
        checkpoint_path = checkpoint_artifact.download()
        checkpoint_file = f"{checkpoint_path}/{model_id}.pth"

        # Override the wandb configs if config paths are provided.
        if args.base_config_file:
            base_config_file = args.base_config_file
            print("Overriding the wandb task config file with "
                  "the base_config_file arg")
        else:
            base_config_file = f"{config_path}/{args.amp_task_name}.yaml"
        if args.train_config_file:
            train_config_file = args.train_config_file
            print("Overriding the wandb training config file with "
                  "the train_config_file arg")
        else:
            train_config_file = f"{config_path}/{args.amp_task_name}PPO.yaml"
    else:
        checkpoint_file = args.checkpoint_file
        # For the config files use the args if provided. Otherwise, use
        # the default ones.
        if args.base_config_file:
            base_config_file = args.base_config_file
        else:
            base_config_file = os.path.join(os.getcwd(), f"cfg/task/{args.amp_task_name}.yaml")
        if args.train_config_file:
            train_config_file = args.train_config_file
        else:
            train_config_file = os.path.join(os.getcwd(), f"cfg/train/{args.amp_task_name}PPO.yaml")
    # Inform the user about the files being used.
    print(f"Using:\n\ttask config: {base_config_file}")
    print(f"\ttrain config: {train_config_file}")
    print(f"\tcheckpoint: {checkpoint_file}")

    if args.test:
        train_config_file = os.path.join(
            os.getcwd(), "cfg/train/PunyoV2AMP_testPPO.yaml")

    with open(os.path.join(os.getcwd(), train_config_file), "r") as stream:
        try:
            train_config = yaml.safe_load(stream)
            print("Train config file was loaded successfully.")
        except yaml.YAMLError as exc:
            print(exc)

    if args.test:
        base_config_file = "cfg/task/PunyoV2AMP_test.yaml"

    with open(os.path.join(os.getcwd(), base_config_file), "r") as stream:
        try:
            task_config = yaml.safe_load(stream)
            print("Task config file was loaded successfully.")
        except yaml.YAMLError as exc:
            print(exc)

    if args.test_success_metrics or args.test:
        task_config["task"]["task_type"]="task_lift"
    task_type = task_config["task"]["task_type"]

    if args.test:
        task_config["task"][task_type]["ampObservation"]= ['robot_dof', 'ee_binary_contact']
    if args.test_success_metrics or args.test:
        task_config["task"]["task_type"]="task_lift"
    task_type = task_config["task"]["task_type"]

    if args.wandb_handle:
        # Update the task config with the demonstration path
        # from wandb
        task_config["task"][task_type]["motion_file_path"] = demo_path

    task_config["env"]["numEnvs"]=args.num_envs
    task_config["sim"]["use_gpu_pipeline"]=True
    task_config["physics_engine"]='physx'
    task_config["sim"]["physx"]["num_threads"]= 4
    task_config["sim"]["physx"]["solver_type"]= 1
    task_config["sim"]["physx"]["num_subscenes"]= 4
    task_config["sim"]["physx"]["use_gpu"]=True
    task_config["env"]["visualizeGoal"] = True

    if not args.test_success_metrics or args.test:
        task_config["task"][task_type]["success_early_termination"] = False

    episode_length = task_config["task"][task_type]["episodeLength"]
    if args.action_mode == "q":
        task_config["env"]["clipActions"] = np.inf
    elif args.action_mode == "delta_q":
        task_config["env"]["clipActions"] = 1

    if args.show_floatie_contact:
        task_config["env"]["visualizeFloatiesContact"] = True
    if args.plot_trajectories:
        task_config["env"]["plotTrajectories"] = True

    if args.action_scale:
        print("\nOverwriting the action scale set in the config:", end=' ')
        print(f"{task_config['env']['actionScale']} -> {args.action_scale}\n")
        task_config["env"]["actionScale"] = args.action_scale

    robot_num_dofs = task_config["env"]["numActions"]

    env = isaacgym_task_map[args.amp_task_name](
        cfg=task_config,
        rl_device="cuda:"+args.gpu_id,
        sim_device="cuda:"+args.gpu_id,
        graphics_device_id=int(args.gpu_id),
        headless=False,
        virtual_screen_capture=False,
        force_render=True,
        action_mode=args.action_mode,
    )
    # Update camera view.
    env.gym.viewer_camera_look_at(env.viewer, None, gymapi.Vec3(11, 6, 2.5), gymapi.Vec3(0, 1.0, -0.5))

    if args.test_success_metrics or args.test:
        q0_robot = np.array(
            [0.2996, 1.0760, 4.5415, 1.2404, 2.6689, 3.8126, 4.1571, 5.9776, 5.1811,
             1.7669, 5.0427, 3.6364, 2.4538, 1.0296])

        print(f"q0_robot: {q0_robot}")
        # Set env initial state.
        for i in range(len(env.punyo_default_dof_poses)):
            env.punyo_default_dof_poses[i] = torch.from_numpy(q0_robot).float().reshape(1,-1).to("cuda:"+args.gpu_id)

    env.reset_idx(torch.arange(args.num_envs, device=env.device))
    env.render()
    obs = env.reset()

    if args.test:
        assert np.all(obs["obs"].to("cpu")[:,:robot_num_dofs].numpy() == q0_robot.astype(np.float32)), "Failure to set the robot initial position."

    sim_time_step = task_config["sim"]["dt"]/task_config["sim"]["substeps"]
    gym_time_step = env.dt
    print(f"gym_time_step: {gym_time_step}, sim_time_step: {sim_time_step}")

    policy = RlGamesPolicy(
        checkpoint_file,
        task_config,
        train_config,
        env.critic_num_obs,
        env.critic_state_types,
        env.policy_state_types,
        )

    obs_dict = env.policy_obs_to_dict(obs["obs"])
    action = obs_dict["robot_dof"][0]
    done = False

    if (args.test_success_metrics or args.test):
        max_episodes = episode_length
    else:
        max_episodes = 1e20

    i = 0
    while i < max_episodes:

        if args.debug:
            progress_time = env.progress_buf[0]*gym_time_step
            print(f"progress_time env0: {progress_time}")
            policy_tic = time.time()
            print(f"mask: {env.actor_mask}")

        # The actor_mask masks out the privilege observations.
        obs_cpu = (obs["obs"]*env.actor_mask).to("cpu")
        action_, _, _ = policy.step(obs_cpu)

        if args.action_mode == "delta_q":
            action = action_
        elif args.action_mode == "q":
            action = tensor_clamp(
                action + env.dt * action_ * env.action_scale,
                env.punyo_dof_pos_lb.to("cpu"),
                env.punyo_dof_pos_ub.to("cpu")
            )

        if args.debug:
            print(f"policy_inference_time: {time.time() - policy_tic} sec")
        print(f"step: {i}")

        if args.debug:
            env_tic = time.time()
        if args.test_success_metrics:
            # Force one environment to fail.
            action[args.num_envs - 1] = np.ones(robot_num_dofs) * 0.1
        obs, _, done_, info = env.step(torch.FloatTensor(action))
        obs_dict = env.policy_obs_to_dict(obs["obs"])
        success_rate = info["success_info"]["success_rate"]
        if "box_pose" in obs_dict.keys():
            print("manipuland xyz: ", obs_dict["box_pose"][0,:3])
            print("manipuland quat: ", obs_dict["box_pose"][0,3:])
        print("success rate: ", success_rate)
        print("success goal distance: ", info["success_info"]["goal_distance"])
        print("success rot distance: ", info["success_info"]["rot_distance"])
        print("success manipuland vel: ", info["success_info"]["manipuland_vel"])

        if args.debug:
            print(f"env_step_elapsed_time: {time.time() - env_tic} sec, gym_time_step: {gym_time_step} sec")
            # This will play the policy step by step.
            input("Press Enter to take a step...")

        env.render()
        i+=1

    if not (args.test or args.test_success_metrics):
        input("Press Enter to exit.")

    if args.test_success_metrics:
        desired_sr = (args.num_envs - 1)/args.num_envs
        assert success_rate > desired_sr and success_rate < 1, f"success ({success_rate}) should be greater than {desired_sr} and less than 1"

if __name__ == "__main__":
    main()
