cd src
python train.py group \
--exp_id crowdhuman_dla34 \
--group_arch 'simple_concat' \
--group_embed_dim 128 \
--gpus 0 --batch_size 10 \
--num_epochs 80 \
--lr_step '50,70' \
--lr 1e-4 \
--data_cfg '../src/lib/cfg/vkist_gta_salsa.json' \
--num_workers 4 \
--val_interval 0 \
--num_sample_positive 16 \
--num_sample_negative 16 \
--load_model "../exp/group/crowdhuman_dla34/model_9.pth" \
--load_model_group "../exp/group/crowdhuman_dla34/group_model_9.pth"
cd ..
