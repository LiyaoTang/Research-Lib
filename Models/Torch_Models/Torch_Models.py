#!/usr/bin/env python
# coding: utf-8
"""
module: classes to construct PyTorch models
"""

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from . import Torch_Modules as torch_modules


""" Siamese Trackers """


class Siam_RPN(nn.Module):
    """
    siam RPN tracker, with different backbone and rpn for choice
    """

    def __init__(self, anchor_generator, config=None):
        super(ModelBuilder, self).__init__()
        if config is None:
            config = {
                'backbone': {
                    'model': torch_modules.MobileNetV2,
                    'kwargs': {'used_layers': [3, 5, 7], 'width_mult': 1.4},
                },
                'neck': {'in_channels': [44, 134, 448], 'out_channels': [256, 256, 256]},
                'rpn': {
                    'model': torch_modules.MultiRPN,
                    'kwargs': {'in_channels': [256, 256, 256], 'anchor_num': 5, 'weighted': False},
                },
            }

        # construct backbone
        cfg = config['backbone']
        backbone = cfg['model']
        self.backbone = backbone(**cfg['kwargs'])

        # construct neck with 1x1 conv (with bn)
        cfg = config['neck']
        self.neck = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            for in_ch, out_ch in zip(cfg['in_channels'], cfg['out_channels'])
        ]
        if len(self.neck) == 1:
            self.neck_forward = self.neck[0]
        else:
            self.neck_forward = lambda feat_list: [nn(f) for nn, f in zip(self.neck, feat_list)]

        # construct rpn
        cfg = config['rpn']
        rpn = cfg['model']
        self.rpn = rpn(**cfg['kwargs'])

        self.config = config
        # self.anchor_generator = anchor_generator # to generate anchors

    def _extract_feature(self, input):
        input_feat = self.backbone(input)
        if self.neck:
            input_feat = self.neck_forward(input_feat)
        return input_feat

    def _regress_box(self, zf, xf):
        logit_cls, logit_loc = self.rpn(zf, xf)

    # def _decode_anchor(self, pred_loc, img_center):
    #     pred_loc = pred_loc.permute(1, 2, 3, 0).contiguous().view(4, -1)  # 
    #     anchors = self.anchors + img_center # shift pre-computed anchors on 

    def inference_onestep(self, x):
        """
        track by searching template on the given region x
        """
        xf = self.backbone(x)
        if self.neck:
            xf = self.neck_forward(xf)
        pred_cls, pred_loc = self.rpn(self.zf, xf)
        # pred_box = self._decode_anchor(pred_loc)

        return {'cls': pred_cls,
                'loc': pred_loc,}

    # def inference(self, xs, z):
    #     """
    #     track over a sequence
    #     """
    #     self._template(z)
    #     rst = []
    #     for x in xs:
    #         rst.append(self.inference_onestep(x))
    #     return rst

    def _log_softmax(self, pred_cls):
        b, a, h, w = pred_cls.size()  # batch, anchor, h, w
        pred_cls = pred_cls.view(b, 2, a // 2, h, w)  # each anchor a 2-channel binary classification
        pred_cls = pred_cls.permute(0, 2, 3, 4, 1).contiguous()  # move cls map to the last
        pred_cls = F.log_softmax(pred_cls, dim=4)  # log(softmax(x)), with better numerical stability
        return pred_cls

    def forward(self, data):
        """ 
        for training (template branch not pruned)
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # forward for pred
        xf = self._extract_feature(search)
        zf = self._extract_feature(template)
        zf = zf[:,:, 4:11, 4:11]  # crop central 7x7

        logit_cls, logit_loc = self.rpn(zf, xf)
        pred_cls = self._log_softmax(logit_cls)
        pred_loc = logit_loc

        # get loss
        cls_loss = select_cross_entropy_loss(pred_cls, label_cls)
        loc_loss = weight_l1_loss(pred_loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = self.config['loss']['cls_weight'] * cls_loss + self.config['loss']['loc_weight'] * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        return outputs
