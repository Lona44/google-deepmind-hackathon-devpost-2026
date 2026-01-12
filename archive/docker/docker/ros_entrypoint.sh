#!/bin/bash
set -e

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Source workspace if built
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Source project workspace if exists
if [ -f /ros2_ws/src/g1_nav/install/setup.bash ]; then
    source /ros2_ws/src/g1_nav/install/setup.bash
fi

# Print ROS2 info
echo "========================================"
echo "G1 ROS2 Nav2 Container"
echo "========================================"
echo "ROS_DISTRO: $ROS_DISTRO"
echo "RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo "========================================"

exec "$@"
