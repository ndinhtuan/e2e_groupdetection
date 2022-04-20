cd src
python test_group_inference.py group \
--group_arch simple_concat \
--load_model /home/giangh/tungch/00_fformation/e2e_groupdetection/exp/group/legacy/model_best.pth \
--load_model_group /home/giangh/tungch/00_fformation/e2e_groupdetection/exp/group/legacy/group_model_best.pth \
--input_width 1024 \
--input_height 768 \
--data_cfg ../src/lib/cfg/fit_uet.json \
--eval_group_ratio 0.666666666 \
--eval_link_threshold 0.9   \
--eval_clustering_algorithm graph_cut \
--eval_highly_connected_rate 0.3 \
--num_workers 8 \
--batch_size 2 \
--eval_save 0
cd ..
