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

RUN git clone https://github.com/julimueller/dtld_parsing.git
RUN git clone https://github.com/bosch-ros-pkg/bstld
RUN git clone https://github.com/lyuwenyu/RT-DETR.git

RUN pip install -r ./bstld/requirements.txt
#RUN pip install -r ./dtld_parsing/requirements.txt

# requirements of rt-detr (we don't use requirements.txt because we want newer pytorch version)
RUN pip install onnx==1.14.0 onnxruntime==1.15.1 pycocotools PyYaml scipy 
RUN python -m pip install --upgrade tensorrt
RUN python3 -m pip install --upgrade tensorrt_lean
RUN python3 -m pip install --upgrade tensorrt_dispatch
RUN python3 -m pip install -U numpy
RUN python3 -m pip install ultralytics

#presentation
RUN python3 -m pip install moviepy natsort
#explainability
RUN python3 -m pip install ttach



ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
