echo "Building publisher from realsense"

cd Examples/ROS/

catkin_

mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
