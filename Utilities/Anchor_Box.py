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
        stride: predefined anchor size = stride^2
        ratios: =height/width => h=stride*sqrt(r), w=stride/sqrt(r)
        scales: a list of scales for same height-width ratio box
    '''

    def __init__(self, stride, ratios, scales, img_center=0, img_size=0):
        super(Anchor_Box, self).__init__()
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.img_center = 0
        self.img_size = 0

        self.anchor_num = len(self.scales) * len(self.ratios)  # each scale with different ratios
        self.anchors = None
        self.generate_anchors()

    def generate_anchors(self):
        '''
        generate anchors based on predefined configuration
        '''
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

    def generate_all_anchors(self, img_center, img_size):
        '''
        generate based on img
        '''
        if self.img_center == img_center and self.img_size == img_size:
            return False
        self.img_center = img_center
        self.img_size = img_size

        a0x = img_center - img_size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = xyxy_to_xywh([x1, y1, x2, y2], float)

        disp_x = np.arange(0, img_size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, img_size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, img_size, img_size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = xywh_to_xyxy([cx, cy, w, h], float)

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w,  h]).astype(np.float32))
        return True
