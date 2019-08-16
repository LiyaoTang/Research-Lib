#!/usr/bin/env python
# coding: utf-8
'''
module: utilities for anchor box processing, including:
    anchor box generation
'''

import numpy as np

class Anchor_Box(object):
    '''
    generate anchor boxes based on 
        stride: predefined anchor size=stride^2 => each anchor mapped back to stride*stride region on input
        ratios: =height/width => h=stride*sqrt(r), w=stride/sqrt(r)
        scales: a list of scales for same height-width ratio box
        size: model output volumn size
        img_center: model input image center
    => prepare for output volume = [size, size, anchor_num*4],
    '''

    def __init__(self, stride, ratios, scales, img_center=0, size=0,
                 iou_thr=(0.3, 0.6), pos_num=16, neg_num=16, total_num=64):
        super(Anchor_Box, self).__init__()
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.img_center = 0
        self.size = 0
        self.iou_thr = sorted((iou_thr, iou_thr) if type(iou_thr) is float else iou_thr)
        assert len(self.iou_thr) == 2
        
        self.pos_num = pos_num  # num of pos anchors
        self.neg_num = neg_num  # num of neg anchors
        self.total_num = total_num  # num of total anchors (including ignored anchors)
        self.anchor_num = len(self.scales) * len(self.ratios)  # each scale with different ratios
        self.anchors = None  # anchors on one location
        self.anchors_volume = None  # anchors for output volume
        
        self._prepare_anchors()
        if img_center > 0 and size > 0:
            self._prepare_anchors_volume(img_center, size)

    def _prepare_anchors(self):
        # prepare anchors based on predefined configuration
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        cnt = 0
        for r in self.ratios:
            sqrt_r = np.sqrt(r)
            ws = int(self.stride / sqrt_r)
            hs = int(self.stride * sqrt_r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[cnt] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5]  # xywh (center at origin) -> xyxy
                cnt += 1

    def _prepare_anchors_volume(self, img_center, size):
        '''
        img_center: model input image center
        size: model output volumn size
        => prepare anchors for a (newly) given input-output
        '''
        if self.img_center == img_center and self.size == size:
            return self.anchors_volume
        self.img_center = img_center
        self.size = size

        # map center of output size back to input size & calc the offset of center
        # => anchor sitting side-by-side, with each anchor covering pixel num = self.stride (on input image)
        a0x = img_center - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)  # starting origin = img_center
        zero_anchors = self.anchors + ori  # shift zero-centered anchors to be centered at img_center
        
        # arr of x1 y1 x2 y2 (of all anchors centered at img_center)
        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]
        
        # extend 2 new axis to -1 dimension => col arr [anchor_num, 1, 1]
        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = xyxy_to_xywh([x1, y1, x2, y2])

        # new axis at first dim => 1 row to broadcast with cx-cy (3 rows)
        # => scatter anchor by stride (onto the input img)
        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride  # [1, 1       , size]
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride  # [1, size,   1     ]

        # shift the center xy: row + col & broadcast
        cx = cx + disp_x  # [anchor_num, 1       , size]
        cy = cy + disp_y  # [anchor_num, size    , 1   ]

        # broadcast
        zero = np.zeros((self.anchor_num, size, size))
        # cx: broadcast to replicate row along axis-0 (duplicate rows & same num clong the col)
        # cy: broadcast to replicate scalar along axis-1 (duplicate cols & same num along the row)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = xywh_to_xyxy([cx, cy, w, h], float)

        # provide both xyxy & xywh - [4, anchor_num, size, size]
        self.anchors_volume = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                               np.stack([cx, cy, w, h]).astype(np.float32))
        return self.anchors_volume

    def _select(position, keep_num=16):
        num = position[0].shape[0]  # position: idx (list) for np arr
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num

    def generate_anchors(self, target, img_center, size, neg=False):
        '''
        generate anchors for given target (label bbox)
        size: output volume size
        '''
        anchor_num = self.anchor_num

        # -1 for ignore; 0 for negative; 1 for positive
        label_cls = np.ones((anchor_num, size, size), dtype=np.int64) - 1  # cooresponding classification for each anchor
        label_loc = np.zeros((4, anchor_num, size, size), dtype=np.float32)  # => 4 * anchor_num * size * size, label location
        label_loc_weight = np.zeros((anchor_num, size, size), dtype=np.float32)  # weight for each anchor

        tcx, tcy, tw, th = xyxy_to_xywh(target) # convert target (label) bbox to xywh

        if neg:  # if retrieving negative sample (pair from diff video)
            # l = size // 2 - 3
            # r = size // 2 + 3 + 1
            # label_cls[:, l:r, l:r] = 0

            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - self.img_center) / self.stride + 0.5))
            cy += int(np.ceil((tcy - self.img_center) / self.stride + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)

            label_cls[:, u:d, l:r] = 0  # crop out the central 7 as neg candidates
            neg, neg_num = self._select(np.where(label_cls == 0), self.neg_num)

            label_cls[:] = -1  # ignore all
            label_cls[neg] = 0  # except for selected negative examples (from the central 7x7 region)

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return label_cls, label_loc, label_loc_weight, overlap  # no location for regression (as all negative or ignored)

        # get prepared anchors volume in xyxy & xywh
        anchor_box, anchor_center = self._prepare_anchors_volume(img_center, size)
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

        label_loc[0] = (tcx - cx) / w  # calc xy offest to anchor (in ratio)
        label_loc[1] = (tcy - cy) / h
        label_loc[2] = np.log(tw / w)  # calc wh in ratio
        label_loc[3] = np.log(th / h)

        overlap = calc_IoU([x1, y1, x2, y2], target)  # bbox overlap - anchor & label box

        pos_idx = np.where(overlap > self.iou_thr[1])  # pos-neg anchor based on IoU thr
        neg_idx = np.where(overlap < self.iou_thr[0])

        pos_idx, pos_num = self._select(pos_idx, self.pos_num)  # select positive with no more than pos_num
        neg_idx, neg_num = self._select(neg_idx, self.total_num - self.pos_num) # total_num: anchor num per image

        label_cls[pos] = 1
        label_loc_weight[pos] = 1 / (pos_num + 1e-6)  # weight for location regression

        label_cls[neg] = 0
        return label_cls, label_loc, label_loc_weight, overlap