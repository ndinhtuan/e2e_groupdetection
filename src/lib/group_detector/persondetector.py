import torch
import numpy as np
from scipy.sparse import csgraph
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat


class GroupDetector(object):

    def __init__(self, opt):
        
        self.opt = opt

        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print("Creating model ...")
        print("Loading main model ...")
        self.main_model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.main_model = load_model(self.main_model, opt.load_main_model)
        self.main_model = self.model.to(opt.device)
        self.main_model.eval()

        #print("Loading group model ...")
        #self.group_model = create_model(opt.arch, opt.heads, opt.head_conv)
        #self.group_model = load_model(self.group_model, opt.load_group_model)
        #self.group_model = self.model.to(opt.device)
        #self.group_model.eval()

    def detect(self, im_blob, img0):
        r"""

        Args:
            im_blob: Tensor image in cuda or cpu
            img0: Original image
        Return:
            List bounding box information
        """

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        
        with torch.no_grad():

            output = self.main_model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]

    def _save_prediction(self, label, preds, dst_file):
        r"""
        Saving bounding box prediction in dst_file

        Args:
            label: Array contains label information
            preds: list bounding box prediction
            dst_file: file to saving bounding box prediction
        """
        pass

    def visualize_with_preds(self, img, preds):
        r"""
        Visualize bounding box prediction from preds on image
        """
        pass

    def visualize(self, img):
        r"""
        Visualize without bounding box prediction
        """
        pass
