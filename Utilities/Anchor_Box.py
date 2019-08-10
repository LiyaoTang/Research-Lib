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
        size: output size
    => prepare for output (pred) volume = [size, size, anchor_num*4],
    '''

    def __init__(self, stride, ratios, scales, center=0, size=0):
        super(Anchor_Box, self).__init__()
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.center = 0
        self.size = 0

        self.anchor_num = len(self.scales) * len(self.ratios)  # each scale with different ratios
        self.anchors = None
        self._prepare_anchors()

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

    def generate_all_anchors(self, center, size):
        '''
        generate anchors for a given output volum
        '''
        if self.center == center and self.size == size:
            return False
        self.center = center
        self.size = size

        a0x = center - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)  # starting origin
        # shift prepared (zero-centered) anchors to be origin-centered (v.s. coord translation)
        zero_anchors = self.anchors + ori
        
        # arr of x1 y1 x2 y2 (of all origin-centered anchors)
        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]
        
        # extend 2 new axis to -1 dimension => col arr [anchor_num, 1, 1]
        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = xyxy_to_xywh([x1, y1, x2, y2], float)

        # new axis at first dim => 1 row to broadcast with cx-cy (3 rows)
        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride # [1, 1       , size]
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride # [1, size,   1     ]

        # shift the center xy: row + col & broadcast
        cx = cx + disp_x # [anchor_num, 1       , size]
        cy = cy + disp_y # [anchor_num, size    , 1   ]

        # broadcast
        zero = np.zeros((self.anchor_num, size, size))
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = xywh_to_xyxy([cx, cy, w, h], float)

        # provide both xyxy & xywh - [4, anchor_num, size, size]
        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w, h]).astype(np.float32))
        return True
