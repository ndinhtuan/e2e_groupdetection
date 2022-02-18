cd src

TRAIN_RAW_PATH="/home/tungch/01_fformation/e2e_groupdetection/data/gta_raw"
VAL_RAW_PATH="/home/tungch/01_fformation/e2e_groupdetection/data/gta_raw_val"
TEST_RAW_PATH="/home/tungch/01_fformation/e2e_groupdetection/data/gta_raw_test"

TRAIN_PROCESSED_PATH="/home/tungch/01_fformation/e2e_groupdetection/data/gta_preprocessed_train"
VAL_PROCESSED_PATH="/home/tungch/01_fformation/e2e_groupdetection/data/gta_preprocessed_val"
TEST_PROCESSED_PATH="/home/tungch/01_fformation/e2e_groupdetection/data/gta_preprocessed_test"

FINAL_PATH="/home/tungch/01_fformation/e2e_groupdetection/data/gta_dataset"
rm -rf $FINAL_PATH
mkdir $FINAL_PATH

python preprocess_format.py --raw_path $TRAIN_RAW_PATH --preprocessed_path $TRAIN_PROCESSED_PATH
python preprocess_format.py --raw_path $VAL_RAW_PATH --preprocessed_path $VAL_PROCESSED_PATH
python preprocess_format.py --raw_path $TEST_RAW_PATH --preprocessed_path $TEST_PROCESSED_PATH

python gen_label_fformation.py \
--src_path $TRAIN_PROCESSED_PATH \
--dst_path $FINAL_PATH \
--phase train \
--cfg_file fformation.train

python gen_label_fformation.py \
--src_path $VAL_PROCESSED_PATH \
--dst_path $FINAL_PATH \
--phase val \
--cfg_file fformation.val

python gen_label_fformation.py \
--src_path $TEST_PROCESSED_PATH \
--dst_path $FINAL_PATH \
--phase test \
--cfg_file fformation.test

cd ..