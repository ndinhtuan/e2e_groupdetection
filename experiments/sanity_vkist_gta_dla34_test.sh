cd src
python test_group.py group \
--group_arch 'attn_concat' \
--load_model /home/tungch/e2e_groupdetection/exp/group/crowdhuman_dla34_test/model_100.pth \
--load_model_group /home/tungch/e2e_groupdetection/exp/group/crowdhuman_dla34_test/group_model_100.pth \
--data_cfg ../src/lib/cfg/sanity_vkist_gta_salsa.json \
--num_workers 8
cd ..
