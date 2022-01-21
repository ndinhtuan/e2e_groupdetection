docker run -t -i --rm --runtime=nvidia \
-v /home/tungch/01_fformation/e2e_groupdetection:/e2e \
-v /home/tungch/01_fformation/e2e_groupdetection:/home/tungch/01_fformation/e2e_groupdetection \
-v /home/tungch/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \
--gpus all e2e_fformation:latest \
sh experiments/vkist_eval_group.sh
