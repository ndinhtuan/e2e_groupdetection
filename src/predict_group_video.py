from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import _init_paths
import argparse
from hcs import labelled_HCS
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
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process
from models.utils import _tranpose_and_gather_feat
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from utils.F1_calc import group_correctness
import copy

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

def clustering(ids, embeds, group_model, link_threshold=0.5, clustering_algorithm="connected_component", highly_connected_rate=0.3):
    
    idx1 = []
    idx2 = []

    num_obj = embeds.shape[0]

    for i in range(num_obj-1):
        for j in range(i+1, num_obj):
            idx1.append(i)
            idx2.append(j)

    embeds1 = torch.Tensor(embeds[idx1])
    embeds2 = torch.Tensor(embeds[idx2])
    
    predict = torch.sigmoid(group_model(embeds1, embeds2, torch.Tensor(embeds)))
    keep = predict > 0.5
    keep = keep.numpy()
    keep_idx1 = np.array(idx1)[keep]
    keep_idx2 = np.array(idx2)[keep]

    graph = [[0 for i in range(num_obj)] for j in range(num_obj)]
    try:
        # for i1, i2 in zip(keep_idx1, keep_idx2):
        #     print("Positive edge", i1, i2)
        #     graph[i1][i2] = 1
        for idx, (i1, i2) in enumerate(zip(idx1, idx2)):
            graph[i1][i2] = int(predict[idx] >= link_threshold)
    except Exception as e:
        print(e)
        import IPython
        IPython.embed()
    
    if num_obj == 0:
        print("GRAPH IS NULL")
        return []

    if clustering_algorithm == "graph_cut":
        cs_graph = csr_matrix(graph)
        n_components, labels = connected_components(csgraph=cs_graph, directed=False, return_labels=True)

        matrixs, groups = {}, {}, 
        for idx, item in enumerate(labels):
            if item not in groups:
                groups[item] = []
            groups[item].append(idx)
        
        for idx in groups:
            aff = []
            for item1 in groups[idx]:
                row = []
                for item2 in groups[idx]:
                    row.append(graph[item1][item2])
                aff.append(row)
            matrixs[idx] = aff
        
        # Loop through connected components, get new strong connected components
        group_strong_cc, group_offsets = {}, {}
        for group_idx in matrixs:
            group_strong_cc[group_idx] = labelled_HCS(
                nx.from_numpy_matrix(np.array(matrixs[group_idx])),
                highly_connected_rate=highly_connected_rate
            )
            group_strong_cc[group_idx] -= 1 # minus 1 to change index offset from 1 to 0
            if group_idx > 0:
                group_offsets[group_idx] = len(np.unique(group_strong_cc[group_idx-1])) + group_offsets[group_idx-1]
            else:
                group_offsets[group_idx] = 0
        
        print("OLD", labels)
        # Update group index
        for group_idx in groups:
            for item_idx, item in enumerate(groups[group_idx]):
                labels[item] = labels[item] + group_offsets[group_idx] + group_strong_cc[group_idx][item_idx]
        print("NEW", labels)
        # import IPython
        # IPython.embed()
    else: # connected component
        cs_graph = csr_matrix(graph)
        n_components, labels = connected_components(csgraph=cs_graph, directed=False, return_labels=True)
    unique_label = np.unique(labels)
    
    res = []
    for label_ in unique_label:
        res.append(list(np.array(ids)[labels==label_]))

    # import IPython
    # IPython.embed()
    return res, graph

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
]*100

def draw_prediction(ids, fformations, dets, img, pred_graph, show_group_boxes=True, show_group_links=False):
    id_dict = dict()

    for i, fformation in enumerate(fformations):
        for id_ in fformation:
            id_dict[id_] = i

    if show_group_links or show_group_boxes:
        total_links, total_boxes = 0, 0
        for t in range(len(dets)):
            id_ = ids[t]
            color_id = id_dict[id_]
            x1 = dets[t, 0]
            y1 = dets[t, 1]
            x2 = dets[t, 2]
            y2 = dets[t, 3]

            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            line_start_x, line_start_y = int((x1 + x2) / 2), int(y2)
            
            if show_group_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[color_id], 4)
                cv2.circle(img, (line_start_x, line_start_y), radius=8, thickness=-1, color=COLORS[color_id])
                total_boxes += 1
            if show_group_links:
                for u in range(t+1, len(dets)):
                    if pred_graph[t][u] == 0:
                        continue
                    id_ = ids[t]
                    next_x1, next_y1, next_x2, next_y2, _ = dets[u]
                    next_x1, next_y1, next_x2, next_y2 = int(next_x1), int(next_y1), int(next_x2), int(next_y2)
                    line_end_x, line_end_y = int((next_x1 + next_x2) / 2), int(next_y2)
                    
                    cv2.circle(img, (line_end_x, line_end_y), radius=8, thickness=-1, color=COLORS[color_id])
                    total_links += 1
                    cv2.line(img, (line_start_x, line_start_y), (line_end_x, line_end_y), COLORS[color_id], 1)

        return img

def predict_image(opt, model, group_model, img):
    
    img0 = copy.deepcopy(img)
    img = torch.from_numpy(img).float().to(opt.device)
    img = img.permute((2, 0, 1))
    img = torch.unsqueeze(img, 0)
    output = model(img)[-1]
    origin_shape = img0.shape
    width = origin_shape[1]
    height = origin_shape[0]
    inp_height = opt.input_height
    inp_width = opt.input_width 
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

    detections, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K)
    id_feature = _tranpose_and_gather_feat(id_feature, inds)[0]

    dets = detections[0].unsqueeze(0)
    dets = post_process(opt, dets, meta)
    dets = list(dets.values())[0]
    #dets = merge_outputs(opt, [dets])[1]
    list_id = [i for i in range(len(dets))]
    id_feature = id_feature.cpu().numpy()

    #print(dets.shape, id_feature.shape); exit()
    id_feature = id_feature[:20]
    list_id = list_id[:20]
    dets = dets[:20]
    cluster, graph = clustering(list_id, id_feature, group_model, \
            link_threshold=opt.eval_link_threshold, clustering_algorithm=\
            opt.eval_clustering_algorithm, highly_connected_rate=opt.eval_highly_connected_rate)
    import IPython
    IPython.embed()
    return draw_prediction(list_id, cluster, dets, img0, graph)

def predict_group(opt):
    result = cv2.VideoWriter('filename.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (1088, 608))
    id_img = 0

    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()

    group_model = create_group_model(opt)
    group_model = load_model(group_model, opt.load_model_group)
    #group_model = group_model.to(opt.device)
    group_model.eval()

    video_path = opt.test_video_path
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.resize(frame, (1088, 608))
        frame = cv2.resize(frame, (opt.input_width, opt.input_height))
        frame = predict_image(opt, model, group_model, frame)
        #cv2.imshow("frame", frame)
        result.write(frame)
        print(f"Writting to disk {opt.input_width}x{opt.input_height}")
        os.makedirs("uetvideo_result", exist_ok=True)
        cv2.imwrite("uetvideo_result/{}.png".format(id_img), frame)
        id_img += 1
        if cv2.waitKey(1) == ord('q'):
            break
        exit()

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    with torch.no_grad():
        predict_group(opt)
