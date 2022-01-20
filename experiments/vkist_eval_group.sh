cd src
python test_group.py group \
--load_model /home/tungch/01_fformation/e2e_groupdetection/exp/group/crowdhuman_dla34/model_780.pth \
--load_model_group /home/tungch/01_fformation/e2e_groupdetection/exp/group/crowdhuman_dla34/group_model_780.pth \
--data_cfg ../src/lib/cfg/vkist_gta_salsa.json \
--num_workers 0
cd ..
