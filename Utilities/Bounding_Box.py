#!/usr/bin/env python
# coding: utf-8
"""
module: utilities for bounding box processing, including:
    xyxy <-> xywh,
    IoU, 
    crop, 
"""

import numpy as np


def xyxy_to_xywh_int(xyxy, dtype=int):
    """
    convert [xmin, ymin, xmax, ymax] -> [x-center, y-center, w, h]
    xy in screen coord => x/y as matrix col/row idx
    """
    xc, yc = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2  # xmin, ymin, xmax, ymax
    h = xyxy[3] - xyxy[1]
    w = xyxy[2] - xyxy[0]
    return np.floor([xc, yc, w, h]).astype(dtype)

def xyxy_to_xywh_float(xyxy, dtype=float):
    xc, yc = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2  # xmin, ymin, xmax, ymax
    h = xyxy[3] - xyxy[1]
    w = xyxy[2] - xyxy[0]
    return np.floor([xc, yc, w, h]).astype(dtype)
xyxy_to_xywh = xyxy_to_xywh_int  # alias


def xywh_to_xyxy_int(xywh, dtype=int):
    """
    convert [x-center, y-center, w, h] -> [xmin, ymin, xmax, ymax]
    xywh in screen coord => x-w/y-h as matrix idx-range in col/row axis
    """
    w_2 = xywh[2] / 2
    h_2 = xywh[3] / 2
    # compensate for jitter (grounding due to int conversion in xyxy_to_xywh)
    return np.ceil([xywh[0] - w_2, xywh[1] - h_2, xywh[0] + w_2, xywh[1] + h_2]).astype(dtype)

def xywh_to_xyxy_float(xywh, dtype=float):
    w_2 = xywh[2] / 2
    h_2 = xywh[3] / 2
    return np.array([xywh[0] - w_2, xywh[1] - h_2, xywh[0] + w_2, xywh[1] + h_2], dtype=dtype)
xywh_to_xyxy = xywh_to_xyxy_int  # alias


def calc_overlap_interval(int1, int2):
    """
    calculate the overlaped interval of 2 intervals ([0] for min val, [1] for max val)
    """
    return np.maximum(0, np.minimum(int1[1], int2[1]) - np.maximum(int1[0], int2[0]))

def calc_IoU(box1, box2):
    """
    calculate the intersection over union for 2 xyxy bbox
    (may broadcast to 2 box arr)
    """
    int_x = calc_overlap_interval((box1[0], box1[2]), (box2[0], box2[2]))
    int_y = calc_overlap_interval((box1[1], box1[3]), (box2[1], box2[3]))
    intersection = int_x * int_y
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union

class Bounding_Box(object):
    """
    bounding box class wrapper
    """
    def __init__(self, box_def, def_type='xywh'):
        if def_type == 'xywh':
            self.x_var = [box_def[0] - int(box_def[2] / 2), box_def[0] + int(box_def[2] / 2)]
            self.y_var = [box_def[1] - int(box_def[3] / 2), box_def[1] + int(box_def[3] / 2)]
            self.w = box_def[2]
            self.h = box_def[3]
        elif def_type == 'xyxy':
            self.x_var = [box_def[0], box_def[2]]
            self.y_var = [box_def[1], box_def[3]]
            self.w = box_def[2] - box_def[0]
            self.h = box_def[3] - box_def[1]
        else:
            raise TypeError

    def intersection(self, bbox):
        sec_x = calc_overlap_interval(self.x_var, bbox.x_var)
        sec_y = calc_overlap_interval(self.y_var, bbox.y_var)
        return sec_x * sec_y

    def IoU(self, bbox):
        i = self.intersection(bbox)
        u = self.w * self.h + bbox.w * bbox.h - i
        return i / u


class Det_Bounding_Box(object):
    """
    bounding class with more granularity (e.g. box over points)
    """
    def __init__(self, box_def):
        if type(box_def) is list:
            self.__init_from_list(box_def)
        elif type(box_def) is dict:
            self.__init_from_dict(box_def)
        else:
            raise TypeError('not supported box_def type \"%s\" with value: \"%s\"' % (str(type(box_def)), str(box_def)))

    def __init_from_list(self, def_list):
        assert len(def_list) == 3  # [[4 xy coords], [elem(s) selected], [prob in onehot]]
        
        self.xy = def_list[0]
        self.x = [xy[0] for xy in self.xy]
        self.y = [xy[1] for xy in self.xy]

        self.x_var = (min(self.x), max(self.x))
        self.y_var = (min(self.y), max(self.y))

        # may be larger than the actual size specified by points - due to resolution and other factors
        self.width = self.x_var[1] - self.x_var[0]
        self.height = self.y_var[1] - self.y_var[0]
        self.size = self.width * self.height

        # elem perspectives
        self.elem = set(def_list[1])
        self.elem_size = len(self.elem)
        
        self.prob = def_list[-1]

        # default setting
        self.valid = True
        self.blockage = 0

    def __init_from_dict(self, box_dict):
        def_list = [[(box_dict['xy'][0], box_dict['xy'][1]),
                     (box_dict['xy'][0] + box_dict['width'], box_dict['xy'][1]),
                     (box_dict['xy'][0], box_dict['xy'][1] + box_dict['height']),
                     (box_dict['xy'][0] + box_dict['width'], box_dict['xy'][1] + box_dict['height'])],
                    box_dict['elem'],
                    box_dict['prob']]
        self.__init_from_list(def_list)
        for k, v in box_dict.items():
            if k not in ['xy', 'elem', 'prob', 'width', 'height']:
                setattr(self, k, v)

    def to_polar(self, origin=(0, 0)):
        """
        convert to polar coord given the origin
        """
        complex_xy = [xy[0] - origin[0] + 1j * (xy[1] - origin[1]) for xy in self.xy]
        self.dist = [np.abs(c_xy) for c_xy in complex_xy]
        self.angle = [np.angle(c_xy) for c_xy in complex_xy]

        self.dist_var = (min(self.dist), max(self.dist))
        self.angle_var = (min(self.angle), max(self.angle))

    @staticmethod
    def overlap_interval(int_1, int_2):
        """
        calculate the overlaped interval of 2 intervals ([0] for min, [1] for max)
        """
        return max(0, min(int_1[1], int_2[1]) - max(int_1[0], int_2[0]))

    @staticmethod
    def xy_intersection(box1, box2):
        """
        calculate the intersection of 2 bbox using the cartesian coord
        """
        sec_x = Det_Bounding_Box.overlap_interval(box1.x_var, box2.x_var)
        sec_y = Det_Bounding_Box.overlap_interval(box1.y_var, box2.y_var)
        return sec_x * sec_y

    @staticmethod
    def elem_intersection(box1, box2):
        """
        calculate the intersection of elements contained by bboxes
        """
        return len(set.intersection(box1.elem, box2.elem))