cd src
python train.py group \
--exp_id crowdhuman_simple_concat_salsa \
--group_arch 'simple_concat' \
--group_embed_dim 128 \
--gpus 0 --batch_size 2 \
--num_epochs 1 \
--lr_step '30,100' \
--lr 1e-5 \
--data_cfg '../src/lib/cfg/fit_salsa.json' \
--num_workers 6 \
--val_interval 0 \
--eval_save_path "/home/giangh/tungch/00_fformation/e2e_groupdetection/src/saving_test" \
--num_sample_positive 16 \
--num_sample_negative 16 \
--load_pretrained_model_group "" \
--load_pretrained_model "/home/giangh/tungch/00_fformation/e2e_groupdetection/models/crowdhuman_dla34.pth" \
--kfold 5
cd ..
