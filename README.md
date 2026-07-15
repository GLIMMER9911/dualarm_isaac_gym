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
python train.py
```

