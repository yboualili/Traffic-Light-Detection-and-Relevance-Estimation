podman run -dti --device nvidia.com/gpu=all --network=host --shm-size 20G\
 --mount type=bind,source=$(pwd),target=/workspace/traffic-light-detection/ \
 --mount type=bind,source=/data/BSTLD/,target=/data/BSTLD/ \
 --mount type=bind,source=/data/DTLD/,target=/data/DTLD/ \
 --mount type=bind,source=/home/ws/uwofa/ccng_jpgs/,target=/data/ccng/ \
  localhost/tld /bin/bash

