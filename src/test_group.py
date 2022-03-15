from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2

from sklearn import metrics
from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
from lib.models.model import create_group_model
from models.model import create_model, load_model
from datasets.dataset.jde import DetDataset, collate_fn
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process
from models.utils import _tranpose_and_gather_feat
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from utils.F1_calc import group_correctness

def post_process(opt, dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes)
    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]


def merge_outputs(opt, detections):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > 128:
        kth = len(scores) - 128
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

def clustering(ids, embeds, group_model):
    
    idx1 = []
    idx2 = []

    num_obj = embeds.shape[0]

    for i in range(num_obj-1):
        for j in range(i+1, num_obj):
            idx1.append(i)
            idx2.append(j)

    print("EMBED SHAPE", embeds.shape)
    embeds1 = torch.Tensor(embeds[idx1])
    embeds2 = torch.Tensor(embeds[idx2])
    
    predict = torch.sigmoid(group_model(embeds1, embeds2, torch.Tensor(embeds)))
    keep = predict > 0.5
    keep = keep.numpy()
    keep_idx1 = np.array(idx1)[keep]
    keep_idx2 = np.array(idx2)[keep]

    graph = [[0 for i in range(num_obj)] for j in range(num_obj)]
    try:
        for i1, i2 in zip(keep_idx1, keep_idx2):
            print("Positive edge", i1, i2)
            graph[i1][i2] = 1
    except Exception as e:
        print(e)
        import IPython
        IPython.embed()
    graph = csr_matrix(graph)
    
    if num_obj == 0:
        print("GRAPH IS NULL")
        return []

    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    unique_label = np.unique(labels)
    
    res = []
    for label_ in unique_label:
        res.append(list(np.array(ids)[labels==label_]))

    return res

COLORS = [
    (0,0,0),
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0),
    (255,0,255),
    (0,255,255),
    (192,192,192),
    (128,128,128),
    (128,0,0),
    (128,128,0),
    (0,128,0),
    (128,0,128),
    (0,128,128),
    (0,0,128),
    (64,64,64),
    (64,0,0),
    (64,64,0),
    (0,64,0),
    (64,0,64),
    (0,64,64),
    (0,0,64),
    (192,192,192),
    (192,0,0),
    (192,192,0),
    (0,192,0),
    (192,0,192),
    (0,192,192),
    (0,0,192)
]
def save_group_test(ids, fformations, dets, img, file_path):
    
    id_dict = dict()

    for i, fformation in enumerate(fformations):
        for id_ in fformation:
            id_dict[id_] = i
    print(id_dict, ids)
    boxes = []
    for t in range(len(dets)):

        id_ = ids[t]
        color_id = id_dict[id_]
        print(color_id, "color_id")
        x1 = dets[t, 0]
        y1 = dets[t, 1]
        x2 = dets[t, 2]
        y2 = dets[t, 3]
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

        # cv2.rectangle(img, (x1, y1), (x2, y2), (10+color_id*30, 10+color_id*30, color_id*30), 4)
        boxes.append((x1, x2, y1, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[color_id], 4)
    
    bbox_file_path = file_path + ".bbox.txt"
    print("Saving image in ", file_path)
    cv2.imwrite(file_path, img)
    print("Saving bbox in ", bbox_file_path)
    with open(bbox_file_path, "w") as fout:
        fout.write(str(len(dets))+"\n")
        fout.write(str(dets))

def compute_f1_score_group(preds, targets, ratio=2/3):
    
    preds = [[[f"ID_00{_id}" for _id in pred] for pred in _preds] for _preds in preds]
    targets = [[
        [f"ID_00{_id}" for _id in target] for target in _targets
    ] for _targets in targets]

    avg_results = np.array([0.0,0.0])
    for pred, target in zip(preds, targets):
        correctness = group_correctness(pred, target, ratio, False)
        TP_n, FN_n, FP_n, precision, recall = correctness
        avg_results += np.array([precision, recall])

    avg_results /= len(preds)
    p, r = avg_results[0], avg_results[1]
    if p*r == 0:
        f1_avg = 0
    else:
        f1_avg = float(2)* avg_results[0] * avg_results[1] / (avg_results[0] + avg_results[1])
    print(f"F1: {f1_avg} - precision: {avg_results[0]} - recall: {avg_results[1]}")
    return f1_avg, avg_results[0], avg_results[1]

def test_group(
        opt,
        batch_size=12,
        img_size=(1088, 608),
        iou_thres=0.3,
        print_interval=40,
        save_result=True,
        eval_dump_maxf1=1.00):
    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    nC = 1
    test_path = data_cfg_dict['test']
    dataset_root = data_cfg_dict['root']
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)

    group_model = create_group_model(opt)
    group_model = load_model(group_model, opt.load_model_group)
    model = model.to(opt.device)
    model.eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = DetDataset(dataset_root, test_path, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=opt.num_workers, drop_last=False, collate_fn=collate_fn)
    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)

    gt_fformation_indexs = []
    pred_fformation_indexs = []

    path_saving_dir = "saving_test"
    if not os.path.isdir(path_saving_dir):
        os.makedirs(path_saving_dir)

    id_dir = 0

    # import IPython
    # IPython.embed()

    for batch_i, (imgs, targets, paths, shapes, targets_len, fformation_indexs) in \
            enumerate(dataloader):
        
        # Create ground truth for fformations

        for dict_ in fformation_indexs:
            fformation_index = []
            for k, v in dict_.items():
                fformation_index.append(v)

            gt_fformation_indexs.append(fformation_index)

        t = time.time()
        #seen += batch_size

        output = model(imgs.cuda())[-1]
        origin_shape = shapes[0]
        width = origin_shape[1]
        height = origin_shape[0]
        inp_height = img_size[1]
        inp_width = img_size[0]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // opt.down_ratio,
                'out_width': inp_width // opt.down_ratio}
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg'] if opt.reg_offset else None
        id_feature = output['id'] # (batch_size x embedding_dim x (width/down_ratio) x (height/down_ratio))
        id_feature = F.normalize(id_feature, dim=1)

        opt.K = 200
        detections, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K) # (batch_size x max_object x 6) , (batch_size x max_object)
        id_feature = _tranpose_and_gather_feat(id_feature, inds) # (batch_size x max_object x embedding_dim)

        # id_feature = id_feature.squeeze(0)
        id_feature = id_feature.cpu().numpy()


        # Compute average precision for each sample
        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
        for si, labels in enumerate(targets):
            seen += 1
            #path = paths[si]
            #img0 = cv2.imread(path)
            dets = detections[si]
            embeds = id_feature[si]
            fformation_index = fformation_indexs[si]
            dets = dets.unsqueeze(0)
            dets = post_process(opt, dets, meta)
            dets = merge_outputs(opt, [dets])[1]

            #remain_inds = dets[:, 4] > opt.det_thres
            #dets = dets[remain_inds]
            if dets is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # If no labels add number of detections as incorrect
            correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                target_cls = labels[:, 0]

                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = xywh2xyxy(labels[:, 2:6])
                target_boxes[:, 0] *= width
                target_boxes[:, 2] *= width
                target_boxes[:, 1] *= height
                target_boxes[:, 3] *= height

                '''
                path = paths[si]
                img0 = cv2.imread(path)
                img1 = cv2.imread(path)
                for t in range(len(target_boxes)):
                    x1 = target_boxes[t, 0]
                    y1 = target_boxes[t, 1]
                    x2 = target_boxes[t, 2]
                    y2 = target_boxes[t, 3]
                    cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.imwrite('gt.jpg', img0)
                for t in range(len(dets)):
                    x1 = dets[t, 0]
                    y1 = dets[t, 1]
                    x2 = dets[t, 2]
                    y2 = dets[t, 3]
                    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.imwrite('pred.jpg', img1)
                abc = ace
                '''

                detected = [] # List of ground truth object id that is detected
                matched = [] # List of dets id
                for i, (*pred_bbox, conf) in enumerate(dets):
                    obj_pred = 0
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                        matched.append(i)
                    else:
                        correct.append(0)
                
                matched_embeds = embeds[matched]
                matched_dets = dets[matched]
                list_detected = [int(i) for i in detected]
                # print("MATCHED", matched)
                # print("LIST DETECTED", list_detected)
                # print("MATCHED EMBEDES", matched_embeds.shape)
                # print("DETS", dets.shape)
                # print("MATCHED DETS", matched_dets.shape)
                cluster = clustering(list_detected, matched_embeds, group_model)
                pred_fformation_indexs.append(cluster)
                if save_result:
                    path = paths[si]
                    img = cv2.imread(path)
                    id_dir += 1
                    gt_cluster = [fformation_index[k] for k in fformation_index]
                    # print("GT CLUSTER", gt_cluster)
                    f1, _, _ = compute_f1_score_group([cluster], [gt_cluster], ratio=opt.eval_group_ratio)
                    if f1 <= eval_dump_maxf1:
                        path = "{}/{}".format(path_saving_dir, f"{f1}F1_{os.path.basename(path)}")
                        print(f"F1={f1}. Saving error image to {path}")
                        # print("SI", si)
                        # print("FF INDEX", fformation_indexs)
                        # print("FULL GT SIZE", len(gt_fformation_indexs))
                        # print("PRED CLUSTER", cluster)
                        # print("GT CLUSTER", gt_fformation_indexs[si])
                        # print("FFORMATION INDEX", fformation_index)
                        # print("FULL GT", gt_fformation_indexs, len(gt_fformation_indexs))
                        # import IPython
                        # IPython.embed()
                        save_group_test(list_detected, cluster, matched_dets, img, path)

            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=correct,
                                              conf=dets[:, 4],
                                              pred_cls=np.zeros_like(dets[:, 4]),  # detections[:, 6]
                                              target_cls=target_cls)

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # Means of all images
            mean_mAP = np.sum(mAPs) / (AP_accum_count + 1E-16)
            mean_R = np.sum(mR) / (AP_accum_count + 1E-16)
            mean_P = np.sum(mP) / (AP_accum_count + 1E-16)

        if batch_i % print_interval == 0:
            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') %
                  (seen, dataloader.dataset.nF, mean_P, mean_R, mean_mAP, time.time() - t))

    print("pred_fformation_indexs: ", pred_fformation_indexs, len(pred_fformation_indexs))
    print("gt_fformation_indexs: ", gt_fformation_indexs, len(gt_fformation_indexs))
    print("F1 score group: ")
    compute_f1_score_group(pred_fformation_indexs, gt_fformation_indexs, ratio=opt.eval_group_ratio)
    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))

    print('AP: %-.4f\n\n' % (AP_accum[0] / (AP_accum_count[0] + 1E-16)))
    print("AP ACCUM", AP_accum)
    print("mAP/R/P", mean_mAP, mean_R, mean_P)

    return mean_mAP, mean_R, mean_P

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    with torch.no_grad():
        map = test_group(opt, batch_size=opt.batch_size, save_result=opt.eval_save, eval_dump_maxf1=opt.eval_dump_maxf1)
