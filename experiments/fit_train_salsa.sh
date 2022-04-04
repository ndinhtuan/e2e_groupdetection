cd src
python train.py group \
--exp_id crowdhuman_simple_concat_salsa \
--group_arch 'simple_concat' \
--group_embed_dim 128 \
--gpus 0 --batch_size 4 \
--num_epochs 150 \
--lr_step '30,100' \
--lr 1e-5 \
--data_cfg '../src/lib/cfg/fit_salsa.json' \
--num_workers 6 \
--val_interval 0 \
--num_sample_positive 16 \
--num_sample_negative 16 \
--load_model "/home/giangh/tungch/00_fformation/e2e_groupdetection/exp/group/crowdhuman_simple_concat_salsa/model_best.pth" \
--load_model_group "/home/giangh/tungch/00_fformation/e2e_groupdetection/exp/group/crowdhuman_simple_concat_salsa/group_model_best.pth"
cd ..
