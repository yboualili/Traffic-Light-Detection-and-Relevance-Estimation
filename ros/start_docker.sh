# test environment
podman run -d -it --net=host localhost/rosbagconverter /opt/entrypoint_ros.sh
# test environment, only works on ada
podman run -d -it --net=host --device nvidia.com/gpu=all --mount type=bind,source=$(pwd),target=/workspace/ros/ \
 --mount type=bind,source=/pool/NikolaiPolley/datasets/CCNG-Rosbags/,target=/workspace/rosbags/ \
  localhost/rosbagconverter /bin/bash -c "echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc && source ~/.bashrc && exec /bin/bash"
