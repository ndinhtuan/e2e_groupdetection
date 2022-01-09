import numpy as np
import os 
import glob
import cv2

def draw_annotation(data_dir, ext="jpg"):
    
    list_img_paths = glob.glob("{}/*.{}".format(data_dir, ext))
    
    for img_path in list_img_paths:

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        annotation_path = img_path.replace(ext, "txt")
        label = np.loadtxt(annotation_path, dtype=np.float64)
        print(label)
        
        for l_ in label:
            bbox = l_[2:6]
            bbox[0] *= w
            bbox[1] *= h
            bbox[2] *= w
            bbox[3] *= h

            w_ = bbox[2]
            h_ = bbox[3]
            
            x, y = bbox[0], bbox[1]
            x1, y1 = x - w_/2, y - h_/2
            x2, y2 = x + w_/2, y + h_/2
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)

def draw_prediction(data_dir, pred_dir, ext="jpg"):
    
    list_img_paths = glob.glob("{}/*.{}".format(data_dir, ext))
    
    for img_path in list_img_paths:

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        name = img_path.split("/")[-1].split(".")[0]
        pred_path = os.path.join(pred_dir, "{}.txt".format(name))
        label = np.loadtxt(pred_path, delimiter=',', dtype=np.float64)
        print(label)
        
        for l_ in label:
            bbox = l_[2:6]

            w_ = bbox[2]
            h_ = bbox[3]
            
            x, y = bbox[0], bbox[1]
            x1, y1 = x , y 
            x2, y2 = x + w_, y + h_
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite("pred_{}.png".format(name), img)
        cv2.waitKey(10)

def draw_gt(data_dir, gt_dir, ext="jpg"):
    
    list_img_paths = glob.glob("{}/*.{}".format(data_dir, ext))
    
    for img_path in list_img_paths:

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        name = img_path.split("/")[-1].split(".")[0]
        gt_path = os.path.join(gt_dir, "{}.txt".format(name))
        label = np.loadtxt(gt_path, dtype=np.float64)
        print(label)
        
        for l_ in label:
            bbox = l_[1:]

            w_ = bbox[2]
            h_ = bbox[3]
            
            x, y = bbox[0], bbox[1]
            x1, y1 = x , y 
            x2, y2 = x + w_, y + h_
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite("gt_{}.png".format(name), img)
        cv2.waitKey(10)

if __name__=="__main__":
    draw_gt("/data/tuannd/fformation/sample_data", "/data/tuannd/fformation/label_detection")
    #draw_prediction("/data/tuannd/fformation/sample_data", "/data/tuannd/fformation/data/outputs/fairmot_mot17/sample_data")
