cd src
python train.py group \
--exp_id crowdhuman_simple_concat_salsa \
--group_arch 'simple_concat' \
--group_embed_dim 128 \
--gpus 0 --batch_size 2 \
--num_epochs 150 \
--lr_step '50,100' \
--lr 1e-4 \
--data_cfg '../src/lib/cfg/fit_salsa.json' \
--num_workers 6 \
--val_interval 0 \
--num_sample_positive 16 \
--num_sample_negative 16 \
--load_model "" \
--load_model_group ""
cd ..
