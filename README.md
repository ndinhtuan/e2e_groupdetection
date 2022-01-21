## Installation
* Clone this repo, and we'll call the directory that you cloned as ${FAIRMOT_ROOT}
* Install dependencies. We use python 3.8 and pytorch >= 1.7.0
```
conda create -n FairMOT
conda activate FairMOT
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
cd ${FAIRMOT_ROOT}
pip install cython
pip install -r requirements.txt
```
* We use [DCNv2_pytorch_1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7) in our backbone network (pytorch_1.7 branch). Previous versions can be found in [DCNv2](https://github.com/CharlesShang/DCNv2).
```
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
```
* In order to run the code for demos, you also need to install [ffmpeg](https://www.ffmpeg.org/).

## Data preparation

```
python src/gen_label_fformation.py
```

## Training

### Training on host
```
sh experiments/gta_dla34.sh
```

### Training on docker
For the first time, build the image
```docker build -t e2e_fformation:latest . < Dockerfile```

After the image has been built, run training/testing script with
```
docker run -t -i --rm --runtime=nvidia -v /home/tungch/01_fformation/e2e_groupdetection:/e2e -v /home/tungch/01_fformation/e2e_groupdetection:/home/tungch/01_fformation/e2e_groupdetection -v /home/tungch/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints --gpus all e2e_fformation:latest sh experiments/vkist_gta_dla34.sh
```

## Evaluation 

### Evaluation for detection 
We utilize github repository https://github.com/rafaelpadilla/Object-Detection-Metrics, so we need to convert prediction and ground truth to the right format. 

1. Step 1: Run detection module

```
python src/detect.py group
```
You should change path to source and destination directory, after running this code, we have prediction file with <left> <top> <width> <height> format. 

2. Similarly, we need to convert format of ground truth of dataset GTA-SALSA to <left> <top> <width> <height> :

```
python src/gen_label_detection.py
```
We also need change source and destimation path in this file to generate in our own environment.

Or, we can use the evaluation inside project:

```
python test_det.py group
```
