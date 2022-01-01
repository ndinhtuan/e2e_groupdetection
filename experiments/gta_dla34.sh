cd src
python train.py group --exp_id crowdhuman_dla34 --gpus 0 --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/gta_salsa.json'
cd ..
