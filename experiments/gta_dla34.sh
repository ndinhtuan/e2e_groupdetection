cd src
python train.py group --exp_id crowdhuman_dla34 --gpus "1" --batch_size 30 --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/gta_salsa.json' --num_workers 0
cd ..
