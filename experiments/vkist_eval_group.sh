cd src
python test_group.py group \
--load_model /root/workspace/01_fformation/e2e_groupdetection/exp/group/crowdhuman_dla34/model_last.pth \
--load_model_group /root/workspace/01_fformation/e2e_groupdetection/exp/group/crowdhuman_dla34/group_model_last.pth \
--data_cfg ../src/lib/cfg/vkist_gta_salsa.json \
--num_workers 8
cd ..
