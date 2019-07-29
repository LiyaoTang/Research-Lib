#!/usr/bin/env python
# coding: utf-8
'''
module: utilities for data processing, including:
    handle bbox: IoU, crop, xyxy <-> xywh
    generate anchors
'''

def xyxy_to_xywh(xyxy):
    '''
    convert [xmin, ymin, xmax, ymax] -> [x-center, y-center, w, h]
    xy in screen coord => x/y as matrix col/row idx
    '''
    xc, yc = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)  # xmin, ymin, xmax, ymax
    h = xyxy[3] - xyxy[1]
    w = xyxy[2] - xyxy[0]
    return np.array([xc, yc, w, h])

def xywh_to_xyxy(xywh):
    '''
    convert [x-center, y-center, w, h] -> [xmin, ymin, xmax, ymax]
    xywh in screen coord => x-w/y-h as matrix idx-range in col/row axis
    '''
    w_2 = int(xywh[2] / 2)
    h_2 = int(xywh[3] / 2)
    return np.array([xywh[0] - w_2, xywh[1] - h_2, xywh[0] + w_2, xywh[1] + h_2])


class Anchors:
    '''
    generate anchors
    '''

    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = 0
        self.size = 0

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        '''
        generate anchors based on predefined configuration
        '''
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size*1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w*0.5, -h*0.5, w*0.5, h*0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        '''
        im_c: image center
        size: image size
        '''
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w,  h]).astype(np.float32))
        return True
