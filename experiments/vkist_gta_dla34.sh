cd src
python train.py group \
--exp_id crowdhuman_dla34 \
--group_arch 'simple_concat' \
--group_embed_dim 134 \
--gpus 2 --batch_size 2 \
--num_epochs 60 \
--lr_step '50' \
--data_cfg '../src/lib/cfg/vkist_gta_salsa.json' \
--num_workers 0 \
--load_model "../exp/legacy/model_last_13012021.pth" \
--load_model_group "" \
--val_interval 0
cd ..
