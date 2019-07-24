#!/usr/bin/env python
# coding: utf-8
'''
module: base class to construct tracker ready for product-level use
'''

class Base_Tracker(object):
    def __init__(self, model_path, gpu_id):
        raise NotImplementedError

    def track_init(self, image, unique_id, starting_box):
        '''
        initialize the tracker with 1st frame image
        '''
        raise NotImplementedError
    
    def track(self, image_paths, starting_box, unique_id):
        '''
        perform tracking on a list of paths to image (as video frames), with a single starting box
        (no on-the-fly fix)
        '''
        raise NotImplementedError
    
    def track_onestep(self, image, unique_id):
        '''
        given current frame & target ID, produce the tracking result for target at current frame
        '''
        raise NotImplementedError

    def track_fix(self, image, unique_id, label_box):
        '''
        fix the result of frame t-1, before tracking on frame t
        '''
        raise NotImplementedError