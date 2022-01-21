cd src
python train.py group \
--exp_id crowdhuman_dla34 \
--group_arch 'simple_concat' \
--group_embed_dim 128 \
--gpus 0 --batch_size 1 \
--num_epochs 500 \
--lr_step '50,100' \
--lr 1e-5 \
--data_cfg '../src/lib/cfg/vkist_gta_salsa.json' \
--num_workers 0 \
--val_interval 0 \
--num_sample_positive 16 \
--num_sample_negative 62 \
--load_model "../exp/legacy/model_last_13012021.pth" \
--load_model_group "../exp/legacy/group_model_last_13012021.pth"
cd ..
