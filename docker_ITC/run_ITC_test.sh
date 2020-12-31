# nvidia-docker run --rm -it -v /home/ytpc2019b/catkin_ws/src/ros_start/scripts:/home dockerfile:latest /bin/bash
xhost +local:user
    NV_GPU='3' nvidia-docker run -it \
    --shm-size=256G \
    --env=DISPLAY=$DISPLAY \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="QT_X11_NO_MITSHM=1" \
    --rm \
    -p 8097:8097 \
    -v /home/akada/home/projects:/home \
    -v /raid:/data \
    --net host \
    itc_t2net:latest \