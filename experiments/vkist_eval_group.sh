cd src
python test_group.py group \
--group_arch simple_average \
--load_model /home/tungch/e2e_groupdetection/exp/group/crowdhuman_simple_concat/model_best.pth \
--load_model_group /home/tungch/e2e_groupdetection/exp/group/crowdhuman_simple_concat/group_model_best.pth \
--data_cfg ../src/lib/cfg/vkist_gta_salsa.json \
--num_workers 4 \
--batch_size 4
cd ..
