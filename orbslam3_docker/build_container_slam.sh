# UI permisions
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

xhost +local:docker

docker pull gdbk1124/orbslam3_docker_tcp:latest

# Remove existing container
#docker rm -f orbslam3 &>/dev/null
#[ -d "ORB_SLAM3" ] && sudo rm -rf ORB_SLAM3 && mkdir ORB_SLAM3

docker run -d --name intrepid_orb3 -it gdbk1124/orbslam3_docker_tcp:latest

# Create a new container
docker run -td --privileged --net=host --ipc=host \
    --name="intrepid_orb3" \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e "XAUTHORITY=$XAUTH" \
    -e ROS_IP=127.0.0.1 \
    --cap-add=SYS_PTRACE \
    -v `pwd`/Datasets:/Datasets \
    -v /etc/group:/etc/group:ro \
    -v `pwd`/ORB_SLAM3:/ORB_SLAM3 \
    gdbk1124/orbslam3_docker_tcp bash
    
# Git pull orbslam and compile
echo "__Copy orbslam to the container and compile"
#docker exec -it intrepid_orb3 bash -i -c "git clone -b orbslam3 https://gitlab.lrz.de/intrepid_EMM/bim-based-vi_slam.git /ORB_SLAM3 && cd /ORB_SLAM3 && chmod +x build.sh && ./build.sh "

cd ..
git checkout orbslam3
docker cp . intrepid_orb3:/ORB_SLAM3

# Compile ORBSLAM3-ROS
echo "__Compile ORBSLAM3-ROS"
docker exec -it intrepid_orb3 bash -i -c "echo 'ROS_PACKAGE_PATH=/opt/ros/melodic/share:/ORB_SLAM3/Examples/ROS'>>~/.bashrc && source ~/.bashrc && cd /ORB_SLAM3 && chmod +x build_ros.sh && ./build_ros.sh"
