BASE_DIR="/root/workspace/01_fformation/e2e_groupdetection"
docker run -t -i --rm --runtime=nvidia \
-v $BASE_DIR:/e2e \
-v $BASE_DIR:/home/tungch/01_fformation/e2e_groupdetection \
-v ~/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \
--gpus all e2e_fformation:latest \
sh experiments/vkist_gta_dla34.sh
