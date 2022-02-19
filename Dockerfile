# docker build -t e2e_fformation:latest . < Dockerfile
# docker run -t -i --rm --runtime=nvidia -v /home/tungch/01_fformation/e2e_groupdetection:/e2e -v /home/tungch/01_fformation/e2e_groupdetection:/home/tungch/01_fformation/e2e_groupdetection -v /home/tungch/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints --gpus all e2e_fformation:latest sh experiments/vkist_gta_dla34.sh
FROM nvidia/cuda:11.0.3-devel
WORKDIR /e2e

RUN apt-get update -y \
    && DEBIAN_FRONTEND="noninteractive" apt-get install build-essential python3-opencv python3-tk  -y \
    && apt-get install -y python3 python3-pip -y \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/cu110/torch_stable.html
RUN pip3 install cython

COPY requirements.txt requirements-docker.txt
COPY DCNv2_latest DCNv2_latest

RUN pip3 install -r /e2e/requirements-docker.txt
RUN cd /e2e/DCNv2_latest && ./make.sh