import cv2
import numpy as np
import os
import glob
import shutil

#def convert_
def write_results_score(filename, results):

    save_format = '{class_id} {x1} {y1} {w} {h}\n'
    with open(filename, 'w') as f:
        for c_xywh in results:
            class_id, x1, y1, w, h = c_xywh
            class_id = 0
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            line = save_format.format(class_id=class_id, x1=x1, y1=y1, w=w, h=h)
            f.write(line)
    print('save results to {}'.format(filename))

def gen_label_detection(src_path, dst_path, ext="jpg"):
    
    r"""
    Generate detection lable following format <class_name> <left> <top> <width> <height> 
    normalize in [0,1]
    With class_name is equal 0, class of human
    """
    if not os.path.isdir(src_path):
        print("Cannot go to {}.".format(src_path))

    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)

    list_path_img = glob.glob("{}/*.{}".format(src_path, ext))
    
    for path_img in list_path_img:

        name_file = path_img.split("/")[-1].split('.')[0]
        dst_label_path = "{}/{}.txt".format(dst_path, name_file)
        src_label_path = "{}/{}.txt".format(src_path, name_file)

        label0 = np.loadtxt(src_label_path, dtype=np.float32).reshape(-1, 20)
        label = label0[:, :6]
        num_object = label.shape[0]
        img = cv2.imread(path_img)
        h, w, _ = img.shape
        xyxys = np.zeros((num_object, 4), dtype=np.int32)
        xyxys[:, 0] = label[:, 2]*w
        xyxys[:, 1] = label[:, 3]*h
        xyxys[:, 2] = label[:, 4]*w
        xyxys[:, 3] = label[:, 5]*h

        xyxys[:, 0] = xyxys[:, 0] - xyxys[:, 2]/2
        xyxys[:, 1] = xyxys[:, 1] - xyxys[:, 3]/2

        res = np.zeros((num_object, 5))
        res[:, 1:5] = xyxys

        write_results_score(dst_label_path, res)

if __name__=="__main__":
    
    SRC_PATH = "/data/tuannd/fformation/sample_data"
    DST_PATH = "/data/tuannd/fformation/label_detection"
    gen_label_detection(SRC_PATH, DST_PATH)
