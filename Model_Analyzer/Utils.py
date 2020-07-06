#!/usr/bin/env python
# coding: utf-8
"""
module: handy utilities for metric calculation
"""

import numpy as np

def IoU_from_confusions(confusions):
    """
    computes IoU from multiple nested confusion matrices
    params:
        confusions: ([..., n_c, n_c], np.int32) - can be any dimension, with the confusion matrices described by
            the last two axes, where n_c = number of classes
    return:
        IoU: ([..., n_c], np.float32) - per-class IoU score for each confusion matrix 
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU
    return IoU