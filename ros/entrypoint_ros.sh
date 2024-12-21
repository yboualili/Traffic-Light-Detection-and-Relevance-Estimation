#!/bin/bash

source /opt/ros/noetic/setup.sh

echo "Starting ROS Core ..."
/opt/ros/noetic/bin/roscore &

echo "Waiting for a bit ..."
sleep 5

echo "Launching Websocket bridge ..."
/opt/ros/noetic/bin/roslaunch rosbridge_server rosbridge_websocket.launch