import os
import glob
import shutil

def gen_fformation_label(src_path, dst_path, config_file, ext="jpg"):
    
    label_folder = "labels_with_ids"
    img_folder = "images"

    if not os.path.isdir(src_path):
        print("Cannot go to {}.".format(src_path))

    if not os.path.isdir(dst_path):
        print("Cannot go to {}.".format(src_path))

    if os.path.isdir("{}/{}".format(dst_path, label_folder)) or 
            os.path.isdir("{}/{}".format(dst_path, img_folder)):

        print("Check image and label file in {}.".format(dst_path))

    list_path_img = glob.glob("{}/*.{}".format(src_path, ext))
    
    with open("{}/{}".format(dst_path, config_file)) as file_:

        for path_img in list_path_img:

            name_file = path_img.split("/")[-1].split('.')[0]
            dst_img_path = "{}/{}/{}.{}".format(dst_path, img_folder, name_file, ext)
            dst_label_path = "{}/{}/{}.txt".format(dst_path, label_folder, name_file)
            src_label_path = "{}/{}.txt".format(dst_path, name_file)
            
            file_.write("{}\n".format(dst_img_path))
            shutil.copyfile(path_img, dst_img_path)
            shutil.copyfile(src_label_path, dst_label_path)

if __name__=="__main__":
    
    SRC_PATH = "/data/tuannd/fformation/sample_data"
    DST_PATH = "/data/tuannd/fformation/gta_dataset"
    CFG_FILE = "fformation.train"
    gen_fformation_label(SRC_PATH, DST_PATH, CFG_FILE)
