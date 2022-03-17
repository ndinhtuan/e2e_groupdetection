cd src
python test_group.py group \
--group_arch simple_concat \
--load_model /home/giangh/tungch/00_fformation/e2e_groupdetection/exp/group/crowdhuman_simple_concat_salsa/model_best.pth \
--load_model_group /home/giangh/tungch/00_fformation/e2e_groupdetection/exp/group/crowdhuman_simple_concat_salsa/group_model_best.pth \
--data_cfg ../src/lib/cfg/fit_salsa.json \
--eval_group_ratio 0.666666666 \
--eval_link_threshold 0.95   \
--eval_clustering_algorithm graph_cut \
--eval_highly_connected_rate 0.3 \
--num_workers 8 \
--batch_size 2 \
--eval_save 0
cd ..
