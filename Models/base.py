#!/usr/bin/env python
# coding: utf-8
'''
module: base class to construct model ready for product-level use
'''

class Base_Model(object):
    def __init__(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError


class Base_Tracker(Base_Model):
    def __init__(self, model_path, gpu_id):
        raise NotImplementedError

    def track_init(self, image, starting_box, track_id=None):
        '''
        initialize the tracker with 1st frame image
        '''
        raise NotImplementedError
    
    def track(self, img_list, starting_box, track_id=None):
        '''
        perform tracking on a list of images / paths to image (as video frames), with a single starting box
        (no on-the-fly fix)
        '''
        raise NotImplementedError
    
    def track_onestep(self, image, track_id):
        '''
        given current frame & target ID, produce the tracking result for target at current frame
        '''
        raise NotImplementedError

    def track_fix(self, image, label_box, track_id):
        '''
        fix the result of frame t-1, before tracking on frame t
        '''
        raise NotImplementedError

    def inference(self, img_list, starting_box, track_id=None):
        '''
        inference <=> track
        '''
        return self.track(img_list, starting_box, track_id)