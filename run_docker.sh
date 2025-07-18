docker build -t sig25_jsgc .

docker run \
    --rm \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD:/host \
    -v /usr/share/nvidia:/usr/share/nvidia \
    --network=host -e NVIDIA_DRIVER_CAPABILITIES=all \
    --privileged \
    --runtime=nvidia \
    -v ${PWD}/codes_for_bsdf:/codes \
    --gpus all \
    -it sig25_jsgc /bin/bash;
