from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit
from lib.models.networks.group.sampling import pair_sampling

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class GroupDetLoss(torch.nn.Module):
    def __init__(self, opt, group_model):
        super(GroupDetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        #self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.group_model = group_model 

        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.BCEWithLogitsLoss()
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

        self.number_sample_group_loss = 30

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                group_embed = _tranpose_and_gather_feat(output['id'], batch['ind']) # (batch_size x max_object x embedding_dim)
                group_embed = group_embed[batch['reg_mask'] > 0].contiguous() # (? x embedding_dim)
                group_embed =  F.normalize(group_embed) # (? x embedding_dim)
                id_target = batch['fformation'][batch['reg_mask'] > 0] # (?)

                # import IPython 
                # IPython.embed()
                #id_output = self.classifier(group_embed).contiguous()
                detections, inds = mot_decode(output['hm'], output['wh'], reg=output['reg'], ltrb=opt.ltrb, K=opt.K)
                detection_embed = detections[batch['reg_mask'] > 0]

                # positive sampling
                pos_embeds1, pos_embeds2, pos_id_samples_1, pos_id_samples_2 = \
                    pair_sampling(group_embed, id_target, self.number_sample_group_loss, positive=True
                )

                # negative sampling
                neg_embeds1, neg_embeds2, neg_id_samples_1, neg_id_samples_2 = \
                    pair_sampling(group_embed, id_target, self.number_sample_group_loss, positive=False
                )
                
                # Add detection features
                pos_embeds1 = torch.cat([pos_embeds1, detection_embed[pos_id_samples_1]], dim=-1)
                pos_embeds2 = torch.cat([pos_embeds2, detection_embed[pos_id_samples_2]], dim=-1)
                neg_embeds1 = torch.cat([neg_embeds1, detection_embed[neg_id_samples_1]], dim=-1)
                neg_embeds2 = torch.cat([neg_embeds2, detection_embed[neg_id_samples_2]], dim=-1)

                pos_pred = self.group_model(pos_embeds1, pos_embeds2)
                neg_pred = self.group_model(neg_embeds1, neg_embeds2)

                pos_shape = pos_pred.shape[0]
                neg_shape = neg_pred.shape[0]
                output_shape = min(2*self.number_sample_group_loss, pos_shape+neg_shape)

                preds = torch.zeros(output_shape)
                labels = torch.zeros(output_shape)

                preds[:self.number_sample_group_loss] = pos_pred
                labels[:self.number_sample_group_loss] = torch.ones(self.number_sample_group_loss)
                preds[self.number_sample_group_loss:] = neg_pred

                id_loss += self.IDLoss(preds, labels)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss + 0.2 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats


class GroupDetTrainer(BaseTrainer):
    def __init__(self, opt, dict_model, optimizer=None):
        self.main_model = dict_model["main_model"]
        self.group_model = dict_model["group_model"]
        super(GroupDetTrainer, self).__init__(opt, self.main_model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = GroupDetLoss(opt, self.group_model)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
