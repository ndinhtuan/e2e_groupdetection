cd src
python train.py group \
--exp_id crowdhuman_dla34_test \
--group_arch 'attn_concat' \
--group_embed_dim 128 \
--gpus 0 --batch_size 1 \
--num_epochs 1000 \
--lr_step '900' \
--lr 1e-5 \
--data_cfg '../src/lib/cfg/sanity_vkist_gta_salsa.json' \
--num_workers 0 \
--val_interval 50 \
--num_sample_positive 16 \
--num_sample_negative 16 \
--load_model "/home/tungch/e2e_groupdetection/exp/group/crowdhuman_simple_concat/model_best.pth" \
--load_model_group ""
cd ..
