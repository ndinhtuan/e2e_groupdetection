cd src
python test_group.py group \
--group_arch simple_concat \
--load_model /home/tungch/e2e_groupdetection/exp/group/crowdhuman_simple_concat/model_best.pth \
--load_model_group /home/tungch/e2e_groupdetection/exp/group/crowdhuman_simple_concat/group_model_best.pth \
--data_cfg ../src/lib/cfg/vkist_salsa.json \
--eval_group_ratio 1 \
--num_workers 8 \
--batch_size 8 \
--eval_save 0
cd ..
