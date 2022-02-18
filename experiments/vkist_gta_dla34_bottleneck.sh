cd src
python -m torch.utils.bottleneck train.py group \
--exp_id crowdhuman_dla34 \
--group_arch 'simple_concat' \
--group_embed_dim 128 \
--gpus 0 --batch_size 20 \
--num_epochs 6 \
--lr_step '50' \
--data_cfg '../src/lib/cfg/vkist_gta_salsa.json' \
--num_workers 0 \
--load_model "" \
--load_model_group ""
cd ..
