#!/usr/bin/env python
# coding: utf-8
'''
module: classes to construct PyTorch models
'''

import cv2
import torch
import numpy as np

class Siamese_Tracker(object):
    '''
    base class for siamese tracker
    '''
    def get_subwindow(self, img, xy, model_size, original_size, channel_avg, gpu_id=0):
        '''
        img: bgr based image
        xy: center position
        model_sz: exemplar size
        original_sz: original size
        avg_chans: channel average
        '''
        if isinstance(xy, float):  # xy = [x, y]
            xy = [xy, xy]
        img_shape = img.shape

        context_xmin = np.floor(xy[0] - original_size / 2)
        context_xmax = context_xmin + original_size - 1
        context_ymin = np.floor(xy[1] - original_size / 2)
        context_ymax = context_ymin + original_size - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - img_shape[1] + 1))
        bottom_pad = int(max(0., context_ymax - img_shape[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = img.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = img
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = channel_avg
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = channel_avg
            if left_pad:
                te_im[:, 0:left_pad, :] = channel_avg
            if right_pad:
                te_im[:, c + left_pad:, :] = channel_avg
            img_patch = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        else:
            img_patch = img[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_size, original_size):
            img_patch = cv2.resize(img_patch, (model_size, model_size))
        img_patch = img_patch.transpose(2, 0, 1)
        img_patch = img_patch[np.newaxis, :, :, :]
        img_patch = img_patch.astype(np.float32)
        img_patch = torch.from_numpy(img_patch)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                img_patch = img_patch.cuda()
        return img_patch
