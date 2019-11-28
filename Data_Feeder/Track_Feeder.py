#!/usr/bin/env python
# coding: utf-8
"""
module: data pipeline for tracking
"""

import os
import sys
sys.path.append('../../')

import numpy as np
import Utilities as utils

from .base import Feeder

__all__ = (
    'Track_Re3_Feeder',
    'Track_Siam_Feeder'
)


class Track_Feeder(Feeder):
    """
    feeder to read prepared tracking dataset (e.g. ImageNet VID)
    reference mapping: a ref -> a track (a series of img with one or more object)
    original reference format: [[video_id, frame_id, track_id(obj), class_id, img_h, img_w, xmin, ymin, xmax, ymax]...],
        where video_dir=video_id, img_name=frame_id => img_path=os.path.join(video_id, frame_id)
    implemented useful functions:
        read img (RGB, BGR)
        convert label type (xyxy, xywh)
        encoding bbox onto single img (crop, mask, mesh)
    note: xy-min/max are in window coord => x indexing col & y indexing row
    """

    def __init__(self, data_ref_path, config={}):
        super(Track_Feeder, self).__init__(data_ref_path, class_num=0, class_name=None, use_onehot=True, config=config)
        self._original_refs = None
        self._xyxy_to_xywh = utils.xyxy_to_xywh
        self._xywh_to_xyxy = utils.xywh_to_xyxy
        self._get_global_trackid_from_ref = lambda ref: tuple(ref[[0, 2]])
        self.base_path = config['base_path']

        assert config['img_lib'] in ['cv2']
        assert config['img_order'] in ['RGB', 'BGR']
        self.img_lib = __import__(config['img_lib'], fromlist=[''])
        if (config['img_lib'] == 'cv2' and config['img_order'] == 'BGR') or \
           (config['img_lib'] == 'skimage' and config['img_order'] == 'RGB'):
           self.imread = lambda path: self.img_lib.imread(path)
        else:
            self.imread = lambda path: self.img_lib.imread(path)[:,:,::-1]

        assert config['label_type'] in ['corner', 'center']
        if config['label_type'] == 'corner':
            self._convert_label_type = np.array
            self.revert_label_type = np.array
        else:  # center encoding (xywh)
            self._convert_label_type = self._xyxy_to_xywh
            self.revert_label_type = self._xywh_to_xyxy

        assert config['bbox_encoding'] in ['crop', 'mask', 'mesh']
        if config['bbox_encoding'] == 'crop':
            self._encode_bbox = self._encode_bbox_crop  # encode bbox to both input img & label
            self.decode_bbox = self._decode_bbox_crop  # decode pred to bbox on full image
        elif config['bbox_encoding'] == 'mask':
            self._encode_bbox = self._encode_bbox_mask
            self.decode_bbox = self._decode_bbox_mask
        else:  # mesh encodeing
            self._encode_bbox = self._encode_bbox_mesh_mask
            self.decode_bbox = self._decode_bbox_mask
        self._get_frame_size_from_ref = lambda ref: tuple(ref[[4, 5]].astype(int))

    def _load_original_refs(self):
        # load the prepared original refs
        if self._original_refs is None:
            refs = np.load(self.data_ref_path)
            # sort first by video, then track, then frame
            idx = np.lexsort((refs[:, 1], refs[:, 2], refs[:, 0]))
            refs = refs[idx]
            self._original_refs = refs

    def reset(self):
        """
        reconstruct feeder accordingly
        """
        raise NotImplementedError

    def _get_img(self, ref):  # original ref
        return self.imread(os.path.join(self.base_path, ref[0], ref[1]))

    @staticmethod
    def _get_mask(img_shape, bbox):
        [xmin, ymin, xmax, ymax] = np.clip(bbox, 0, img_shape[[1, 0, 1, 0]]).astype(int)  # the actual region to focus
        mask = np.zeros(shape=img_shape[[0, 1]])
        mask[ymin:ymax, xmin:xmax] = 1
        return mask[...,np.newaxis]

    def _encode_bbox_mask(self, img, prev_box, cur_box):
        img_shape = np.array(img.shape)
        mask = self._get_mask(img_shape, prev_box)
        img = np.concatenate([img, mask], axis=-1)
        return img, cur_box
    
    def _encode_bbox_mesh_mask(self, img, prev_box, cur_box):
        # additionally provide network with pixel location [i,j] at each location
        img_shape = np.array(img.shape)
        mask = self._get_mask(img_shape, prev_box)
        Y, X = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))
        img = np.concatenate([img, mask, Y[..., np.newaxis], X[..., np.newaxis]], axis=-1)
        return img, cur_box

    def _encode_bbox_crop(self, img, crop_region, cur_box, patch_size, pad_val=0):
        # patch: final output (model input)
        # crop_region: xywh, the desired content on original img (may exceed the edge)
        img_shape = np.array(img.shape)
        crop_shape = (crop_region[3], crop_region[2], img_shape[-1]) # row-h, col-w, channel
        img_patch = np.zeros(shape=crop_shape) + pad_val

        # desired & actual crop region (xyxy)
        desired_region = self._xywh_to_xyxy(crop_region)
        actual_region = np.clip(desired_region, 0, img_shape[[1, 0, 1, 0]]).astype(int)

        # convert coord for actual crop region: from img coord to img crop
        xyxy_in_crop = actual_region - desired_region[[0, 1, 0, 1]]

        # crop on original img & resize
        [xmin, ymin, xmax, ymax] = actual_region
        img_patch[xyxy_in_crop[1]:xyxy_in_crop[3], xyxy_in_crop[0]:xyxy_in_crop[2]] = img[ymin:ymax, xmin:xmax]
        img_patch = self.img_lib.resize(img_patch, (patch_size, patch_size))

        # # resize as crop => faster
        # xyxy_in_patch = (xyxy_in_crop / crop_shape[[2, 3, 2, 3]] * patch_size).astype(int)
        # wh_in_patch = [xyxy_in_patch[2] - xyxy_in_patch[0], xyxy_in_patch[3] - xyxy_in_patch[1]]
        # img_patch[xyxy_in_patch[1]:xyxy_in_patch[3], xyxy_in_patch[0]:xyxy_in_patch[2]] = \
        #     self.img_lib.resize(img[ymin:ymax, xmin:xmax], wh_in_patch)

        # convert cur_box (current label) to the crop coord
        label_box = cur_box - desired_region[[0, 1, 0, 1]]  # originated as [x,y,x,y] - [xmin,ymin,xmin,ymin]
        label_box[[0, 2]] = label_box[[0, 2]] / crop_shape[1] * patch_size  # normalized as x / w ratio, then rescale
        label_box[[1, 3]] = label_box[[1, 3]] / crop_shape[0] * patch_size  # normalized as y / h ratio, then rescale
        return img_patch, label_box

    def _decode_bbox_crop(self, crop_region, cur_box):
        # crop_region: xywh
        desired_region = self._xywh_to_xyxy(crop_region)
        box = self.revert_label_type(cur_box)  # as xyxy
        
        box[[0, 2]] = box[[0, 2]] / self.crop_size * crop_region[2]  # recover ratio and rescale according to original img shape
        box[[1, 3]] = box[[1, 3]] / self.crop_size * crop_region[3]
        box = box + desired_region[[0, 1, 0, 1]]  # originated back to img coord
        return box.astype(int)

    def _decode_bbox_mask(self, img, prev_box, cur_box):
        box = self.revert_label_type(cur_box)
        return np.clip(box, 0, np.array(img.shape)[[1, 0, 1, 0]]).astype(int)

    @staticmethod
    def _clip_bbox_from_ref(ref, image_shape):  # original ref
        xmin, ymin, xmax, ymax = np.clip(ref[-4:].astype(int), 0, image_shape[[1, 0, 1, 0]])
        return xmin, ymin, xmax, ymax
    
    def _solve_labelbox_center(self, ref, image_shape):
        xyxy_box = self._clip_bbox_from_ref(ref, image_shape)
        xywh_box = self._xyxy_to_xywh(xyxy_box)
        return xywh_box, xyxy_box

    def _solve_labelbox_corner(self, ref, image_shape):
        xyxy_box = self._clip_bbox_from_ref(ref, image_shape)
        return xyxy_box, xyxy_box

    def encode_bbox_to_img(self, img, prev_bbox):
        img, _ = self._encode_bbox(img, prev_bbox, [0,0,0,0])
        return img


class Track_Re3_Feeder(Track_Feeder):
    """
    feeder for re3 tracker
    original reference format: [[video_id, frame_id, track_id, class_id, img_h, img_w, xmin, ymin, xmax, ymax]...],
        where video_dir=video_id, img_name=frame_id => img_path=os.path.join(video_id, frame_id)    
    """

    def __init__(self, data_ref_path, num_unrolls=None, batch_size=None, img_lib='cv2', config={}):
        config['img_lib'] = img_lib
        config['img_order'] = 'RGB'
        super(Track_Re3_Feeder, self).__init__(data_ref_path, config)
        self.num_unrolls = None
        self.batch_size = None
        self._original_refs = None
        self.crop_size = 227

        assert config['bbox_encoding'] in ['crop', 'mask', 'mesh']
        if config['bbox_encoding'] == 'crop':
            self.patch_size = 277
            self._get_input_label_example = self._get_input_label_example_crop # feed pair of crop
        elif config['bbox_encoding'] == 'mask':
            self._get_input_label_example = self._get_input_label_example_mask # feed full img with mask
        else:  # mesh encodeing
            self._get_input_label_example = self._get_input_label_example_mask

        # construct self.data_refs & record num_unrolls/batch_size, if provided
        if num_unrolls and batch_size:
            self.reset(num_unrolls, batch_size)

    def reset(self, num_unrolls=None, batch_size=None):
        """
        reconstruct feeder according to specified num_unrolls
        """
        if self.num_unrolls != num_unrolls and num_unrolls is not None:
            self.num_unrolls = num_unrolls
        if self.batch_size != batch_size and batch_size is not None:
            self.batch_size = batch_size
        assert self.num_unrolls is not None and self.batch_size is not None
        self._load_data_ref()

    def _load_data_ref(self):
        self._load_original_refs()

        ref_dict = defaultdict(lambda: []) # img size -> [track ref, ...], track_ref=idx in original ref (track len fixed)
        for idx in range(len(self._original_refs) - self.num_unrolls):
            start = self._get_global_trackid_from_ref(self._original_refs[idx])
            end = self._get_global_trackid_from_ref(self._original_refs[idx + self.num_unrolls - 1])
            size = self._get_frame_size_from_ref(self._original_refs[idx])
            if start == end:  # still in the same track
                ref_dict[size].append(idx)  # split into groups based on img size

        # construct data_ref = [batch, ...], each batch = [track_ref, ...] (randomized)
        data_ref = []
        for ref_list in ref_dict.values():
            np.random.shuffle(ref_list)
            batch_num = int(np.ceil(len(ref_list) / self.batch_size))  # at least one batch 
            for i in range(batch_num):
                start = i * self.batch_size
                cur_batch = ref_list[start:start + self.batch_size]
                while len(cur_batch) < self.batch_size:  # fill the last batch with wrapping over
                    cur_batch += ref_list[0:self.batch_size - len(cur_batch)]
                data_ref.append(cur_batch)
        np.random.shuffle(data_ref)
        self.data_ref = data_ref

    def _get_input_label_pair(self, ref):
        # generate a pair of batch
        input_batch = []
        label_batch = []
        for track_ref in ref:  # for ref in current batch
            cur_input, cur_label = self._get_input_label_example(track_ref)
            input_batch.append(cur_input)
            label_batch.append(cur_label)
        return input_batch, label_batch

    def _get_crop_region(self, box):
        x, y, w, h = self._xyxy_to_xywh(box)
        return [x, y, 2 * w, 2 * h]        

    def _encode_bbox_crop(self, img, prev_box, cur_box):
        crop_region = self._get_crop_region(prev_box)  # prev_box from pred/label in xyxy
        return super(self, Track_Re3_Feeder)._encode_bbox_crop(img, crop_region, cur_box, self.patch_size, pad_val=0)

    def _decode_bbox_crop(self, img, prev_box, cur_box):
        crop_region = self._get_crop_region(prev_box)  # prev_box from pred/label in xyxy
        return super(self, Track_Re3_Feeder)._decode_bbox_crop(img, crop_region, cur_box)

    def _get_input_label_example_crop(self, track_ref):
        # generate pair of images of time [t, t-1] as net input at time t
        org_ref = self._original_refs[track_ref:track_ref + self.num_unrolls]  # get original_ref for a track
        if np.random.rand() < self.config['use_inference_prob']:
            return self._get_input_label_from_inference(track_ref)

        input_seq = []
        label_seq = []
        prev_box = None
        prev_input = None
        for r in org_ref:  # for original_ref in a track/video
            cur_img = self._get_img(r)
            cur_box = self._clip_bbox_from_ref(r, np.array(cur_img.shape))  # xyxy box

            if prev_box is None:
                prev_box = cur_box
            cur_input, cur_label = self._encode_bbox(cur_img, prev_box, cur_box)

            if prev_input is None:
                prev_input = cur_input
            input_seq.append((prev_input, cur_input))
            label_seq.append(self._convert_label_type(cur_label))

            prev_box = cur_box
            prev_input = cur_input
        return input_seq, label_seq

    def _get_input_label_example_mask(self, track_ref):
        # generate mask based on label box of time t-1
        org_ref = self._original_refs[track_ref:track_ref + self.num_unrolls]  # get original_ref for a track
        if np.random.rand() < self.config['use_inference_prob']:
            return self._get_input_label_from_inference(track_ref)

        input_seq = []
        label_seq = []
        prev_box = None
        for r in org_ref:  # for original_ref in a track
            cur_img = self._get_img(r)
            cur_box = self._clip_bbox_from_ref(r, np.array(cur_img.shape))  # xyxy box

            if prev_box is None:
                prev_box = cur_box
            cur_input, cur_label = self._encode_bbox(cur_img, prev_box, cur_box)
            prev_box = cur_box
            
            input_seq.append(cur_input)
            label_seq.append(self._convert_label_type(cur_label))
        return input_seq, label_seq

    def _get_input_label_from_inference(self, ref):
        img_seq = []
        box_seq = []
        for r in ref:  # for original_ref in constructed track ref
            cur_img = self._get_img(r)
            cur_box = self._clip_bbox_from_ref(r, np.array(cur_img.shape))  # xyxy box
            img_seq.append(cur_img)
            box_seq.append(cur_box)

        pred_seq = self.config['model'].inference(img_seq, box_seq, self.config['sess'])

        input_seq = []
        label_seq = []
        prev_box = None
        prev_input = None
        for cnt in range(len(ref)):
            img = img_seq[cnt]
            label_box = box_seq[cnt]
            prev_box = box_seq[0] if cnt == 0 else pred_seq[cnt - 1]
            cur_input, cur_label = self._encode_bbox(img, prev_box, label_box)

            if self.config['bbox_encoding'] == 'crop' and prev_input is None:
                cur_input = (cur_input, prev_input)
            input_seq.append(cur_input)
            label_seq.append(cur_label)

            if self.config['bbox_encoding'] == 'crop':
                prev_input = cur_input

        return input_seq, label_seq

    def iterate_track(self):
        """
        iterate over all tracks in dataset as (input_seq, label_seq)
        Warning: should not be used if feeder is currently wrapped by parallel feeder (as feeder states modified)
        """
        _use_inference_prob = self.config['use_inference_prob']
        _num_unrolls = self.num_unrolls

        self.config['use_inference_prob'] = -1
        idx = 0
        track_input, track_label = [], []
        while idx < len(self._original_refs):
            # collect cur track
            start_idx = idx
            vid = list(self._original_refs[idx][[0, 1, 3]])
            while idx < len(self._original_refs) and vid == list(self._original_refs[idx][[0, 1, 3]]):
                idx += 1
            self.num_unrolls = idx - start_idx
            yield self._get_input_label_example(start_idx)

        self.config['use_inference_prob'] = _use_inference_prob
        self.num_unrolls = _num_unrolls


class Track_Siam_Feeder(Track_Feeder):
    """
    feeder for siam tracker (e.g. siam fc, siam rpn)
    original reference format: [[video_id, frame_id, track_id, class_id, img_h, img_w, xmin, ymin, xmax, ymax]...],
        where video_dir=video_id, img_name=frame_id => img_path=os.path.join(video_id, frame_id)    
    frame_range: given frame idx of template(z), compose a positive example with search(x) drawn inside idx+/-frame_range 
    pos_num: the least num/ratio of positive z-x pair inside a batch
    """

    def __init__(self, data_ref_path, frame_range=None, pos_num=0.8, batch_size=None, config={}):
        config['img_lib'] = 'cv2'
        config['img_order'] = 'RBG'
        super(Track_Siam_Feeder, self).__init__(data_ref_path, config)
        self._original_refs = None
        self.ref_dict = None
        self.crop_size = 511

        assert config['bbox_encoding'] in ['crop', 'mask', 'mesh']
        if config['bbox_encoding'] == 'crop':
            self.template_size = config['template']['size']
            self.search_size = config['search']['size']
            self._prepare_template = self._prepare_mask  # feed pair of cropped img
            self._prepare_search = self._prepare_mask
        elif config['bbox_encoding'] == 'mask':
            self._prepare_template = self._prepare_mask  # feed pair of full img with mask
            self._prepare_search = self._prepare_mask
        else:  # mesh encodeing
            self._prepare_template = self._prepare_mask
            self._prepare_search = self._prepare_mask

        # construct self.data_refs & record frame_range-batch_size-pos_num, if provided
        if frame_range and pos_num and batch_size:
            self.reset(frame_range, pos_num, batch_size)

    def reset(self, frame_range=None, pos_num=None, batch_size=None):
        """
        reconstruct feeder according to specified frame_range & batch_size
        """
        if self.frame_range != frame_range and frame_range is not None:
            self.frame_range = frame_range
        if self.batch_size != batch_size and batch_size is not None:
            self.batch_size = batch_size
        if pos_num is float:
            pos_num = int(self.batch_size * pos_num)
        if self.pos_num != pos_num and pos_num is not None:
            self.pos_num = pos_num
        assert self.frame_range is not None and self.batch_size is not None and self.pos_num is int
        self._load_data_ref()

    def _load_data_ref(self):
        self._load_original_refs()
        
        if self.ref_dict is None:
            # img size -> [track ref, ...], track_ref=(start_idx, end_idx) of original ref
            ref_dict = defaultdict(lambda: [])
            start_idx = 0
            start = self._get_global_trackid_from_ref(self._original_refs[start_idx])
            size = self._get_frame_size_from_ref(self._original_refs[start_idx])
            for idx in range(1, len(self._original_refs)):
                end = list(self._original_refs[idx][[0, 1, 3]])
                if start != end:  # encounter new track => record last track & start new track
                    ref_dict[size].append((start_idx, idx))  # split into groups based on img size
                    start, size = update_info(idx)
                    start_idx = idx
            if idx != start_idx:  # finish the last track
                ref_dict[size].append((start_idx, idx))
            self.ref_dict = ref_dict  # needed for neg example retrieval
        
        # construct [batch, ...], batch = [z-x, ..., id], z-x are original refs for positive pairs only, id the idx in ref_dict
        data_ref = []
        for cur_size, ref_list in ref_dict.items():
            example_list = [] # sample examples (z-x pair)
            for start_idx, end_idx in ref_list:
                for z_idx in range(start_idx, end_idx):  # use each img as template for at least once
                    x_idx_min = max(z_idx - self.frame_range, start_idx)
                    x_idx_max = min(z_idx + self.frame_range + 1, end_idx)
                    x_idx = np.random.randint(x_idx_min, x_idx_max)
                    example_list.append((z_idx, x_idx))
            np.random.shuffle(example_list)
            cnt = 0  # form batches (using constructed examples)
            while cnt < len(example_list):
                batch = example_list[cnt:cnt + self.pos_num]
                cnt += self.pos_num
                while len(batch) < self.pos_num:  # finish the last batch
                    batch += example_list[0:self.pos_num - len(batch)]
                batch.append(cur_size)  # to denote the size of each batch
                data_ref.append(batch)
        np.random.shuffle(data_ref)
        self.data_ref = data_ref

    def _get_input_label_pair(self, ref):
        # generate a pair of batch
        input_batch = []
        label_batch = []

        for (z_idx, x_idx) in ref[:-1]:  # for positive z-x pair in current batch
            cur_input, cur_label = self._get_input_label_example(z_idx, x_idx)
            input_batch.append(cur_input)
            label_batch.append(cur_label)
        for _ in range(self.batch_size - self.pos_num):  # sample neg z-x pair
            z_idx, x_idx = self._sample_neg_example(ref[-1])
            cur_input, cur_label = self._get_input_label_example(z_idx, x_idx)
            input_batch.append(cur_input)
            label_batch.append(cur_label)

        return input_batch, label_batch

    def _sample_neg_example(self, size):
        z_track = np.sample(self.ref_dict[size])
        z_idx = np.random.randint(z_track[0], z_track[1])
        x_track = np.sample(self.ref_dict[size])
        while x_track == z_track:
            x_track = np.sample(self.ref_dict[size])
        x_idx = np.random.randint(x_track[0], x_track[1])
        return z_idx, x_idx

    def _get_input_label_example(self, z_idx, x_idx):
        z_ref = self._original_refs[z_idx]
        z_img = self._get_img(z_ref)
        z_box = self._clip_bbox_from_ref(z_ref)

        x_ref = self._original_refs[x_idx]
        x_img = self._get_img(x_ref)
        x_box = self._clip_bbox_from_ref(x_ref)

        template, _  = self._prepare_template(z_img, z_box)
        search, label = self._prepare_search(x_img, x_box)

        return (template, search), label

    def _prepare_template_crop(self, img, box):
        # all use their own box (instead of prev box for cur img in re3)
        # => in case prev box (template) do not contain target in search img (due to large frame range)
        crop_region = self._get_crop_region(box)
        template, _ = super(self, Track_Siam_Feeder)._encode_bbox_crop(img, crop_region, [0, 0, 0, 0],
                                                                       self.template_size,
                                                                       pad_val=np.mean(img, axis=(0, 1)))
        return template, None

    def _prepare_search_crop(self, img, box):
        crop_region = self._get_crop_region(box)
        search, label = super(self, Track_Siam_Feeder)._encode_bbox_crop(img, crop_region, box,
                                                                       self.search_size,
                                                                       pad_val=np.mean(img, axis=(0, 1)))
        return search, label

    def _prepare_mask(self, img, box):
        patch, _ = self._encode_bbox(img, box, None)
        return patch, box

    def iterate_track(self):
        """
        iterate over all tracks in dataset as (template, search imgs, labels)
        """
        for ref_list in self.ref_dict.values():
            track_input, track_label = [], []
            for start_idx, end_idx in ref_list:  # a track
                z_ref = self._original_refs[start_idx]
                z_img = self._get_img(z_ref)
                z_box = self._clip_bbox_from_ref(z_ref)
                template, _ = self._prepare_template(z_img, z_box)

                search_list = []
                label_list = []
                for ref in self._original_refs[start_idx + 1:end_idx]:
                    img = self._get_img(ref)
                    box = self._clip_bbox_from_ref(ref)
                    search, label = self._prepare_search(img, box)
                    search_list.append(search)
                    label_list.append(label)
            yield template, search_list, label_list

    def _get_crop_region(self, box):
        x, y, w, h = self._xyxy_to_xywh(box)
        context_amount = (w + h) / 2
        w += context_amount
        h += context_amount
        norm_sz = np.sqrt(w * h)  # convert box into square region (after extended with some context space)
        return [x, y, norm_sz, norm_sz]