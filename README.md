# dualarm_isaac_gym

## using isaacgym image

```bash
docker images
xhost +local:docker
docker run -it --rm \ 
    --gpus all \
    --network host \
    -v /home/dong/Documents/PhD_Dualarm/dualarm_isaac_gym:/workspace/dualarm_isaac_gym \
    isaacgym:dualarm \
    /bin/bash 
```

or 

```bash
./run_docker.sh

docker exec -it e DISPLAY=$DISPLAY dualarm_isaac_gym_container /bin/bash
```

```
./create_conda_env_rlgpu.sh
```

test 

```bash
cd ./isaacgymenvs/ 

export ISAAC_GYM_PATH=/opt/isaacgym 

python train.py task=DualArmReach headless=True num_envs=64

python train.py task=DualArmReach train=DualArmReachPPO test=True checkpoint="runs/DualArmReach_15-16-50-55/nn/last_DualArmReach_ep_10000_rew_9.01048.pth" num_envs=1 headless=False
```

