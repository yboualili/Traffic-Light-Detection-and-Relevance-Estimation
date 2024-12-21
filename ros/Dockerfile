FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

#ENV PATH=/opt/cmake-3.14.6-Linux-x86_64/bin/:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y \
    #apt-utils \
    git \
    ssh \ 
    #python-is-python3 \ 
    python3-pip \
   libcppunit-dev \
   clang \
   clang-format \
   vim \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN git clone https://github.com/bosch-ros-pkg/bstld
RUN pip install -r ./bstld/requirements.txt

# requirements of rt-detr (we don't use requirements.txt because we want newer pytorch version)
RUN python3 -m pip install -U numpy
RUN python3 -m pip install ultralytics


## ROS

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN sh -c 'echo "deb http://packages.ros.org/ros-testing/ubuntu focal main" > /etc/apt/sources.list.d/ros-testing.list'

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
	ros-noetic-rosbridge-server \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install rospkg pycryptodomex gnupg twisted autobahn tornado pymongo

RUN conda uninstall libtiff -y
RUN conda install libtiff=4.0.8 -y
RUN pip install Pillow==2.2.2
RUN conda install libffi==3.3 -y
RUN pip install torchvision==0.16.0

COPY entrypoint_bash.sh /opt/entrypoint_bash.sh
RUN chmod +x /opt/entrypoint_bash.sh

COPY entrypoint_ros.sh /opt/entrypoint_ros.sh
RUN chmod +x /opt/entrypoint_ros.sh
