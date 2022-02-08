cd src
python train.py group \
--exp_id crowdhuman_dla34 \
--group_arch 'simple_concat' \
--group_embed_dim 128 \
--gpus 0 --batch_size 4 \
--num_epochs 5 \
--num_iters 1 \
--lr_step '50,100' \
--lr 1e-5 \
--data_cfg '../src/lib/cfg/vkist_gta_salsa.json' \
--num_workers 0 \
--val_interval 0 \
--num_sample_positive 16 \
--num_sample_negative 16 \
--load_model "" \
--load_model_group ""
cd ..
