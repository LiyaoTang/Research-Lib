#!/usr/bin/env python
# coding: utf-8
'''
script: construct a npy file for label of ILSVRC2015 detectoin from video (VID) dataset
'''

import os
import cv2
import sys
import time
import glob
import random
import numpy as np
import xml.etree.ElementTree as ET

DEBUG=True

def main(label_type):
    wildcard = '/*/*/' if label_type == 'train' else '/*/'
    dataset_path = '../../../data/ILSVRC2015/'
    annotationPath = dataset_path + 'Annotations/'

    # video Data:       ILSVRC2015/Data       /VID/[train, val, test]/[package]/[snippet ID]/[frame ID].JPEG
    # video annotation: ILSVRC2015/Annotations/VID/[train, val]      /          [snippet ID]/[frame ID].xml
    video_dirs = sorted(glob.glob(annotationPath + 'VID/' + label_type + wildcard)) # all video folders
    total_image_cnt = len(glob.glob(annotationPath + 'VID/' + label_type + wildcard + '*.xml'))

    bboxes = []
    classes = {
            'n01674464': 1,
            'n01662784': 2,
            'n02342885': 3,
            'n04468005': 4,
            'n02509815': 5,
            'n02084071': 6,
            'n01503061': 7,
            'n02324045': 8,
            'n02402425': 9,
            'n02834778': 10,
            'n02419796': 11,
            'n02374451': 12,
            'n04530566': 13,
            'n02118333': 14,
            'n02958343': 15,
            'n02510455': 16,
            'n03790512': 17,
            'n02391049': 18,
            'n02121808': 19,
            'n01726692': 20,
            'n02062744': 21,
            'n02503517': 22,
            'n02691156': 23,
            'n02129165': 24,
            'n02129604': 25,
            'n02355227': 26,
            'n02484322': 27,
            'n02411705': 28,
            'n02924116': 29,
            'n02131653': 30,
            }

    cnt = 0
    for cur_video_dir in (video_dirs):
        path_list = cur_video_dir.strip('/').split('/')
        if label_type == 'train':
            package_id = int(path_list[-2].split('_')[-1])
        else:
            package_id = -1
        video_id = int(path_list[-1].split('_')[-1])
        
        label_path = sorted(glob.glob(cur_video_dir + '*.xml')) # all labels in a video snippet
        image_path = [label.replace('Annotations', 'Data').replace('xml', 'JPEG') for label in label_path] # corresponding images

        if DEBUG:
            track_color = dict()
        for img_p, label_p in zip(image_path, label_path):
            if cnt % 1000 == 0:
                print('cnt %d of %d = %.2f%%' % (cnt, total_image_cnt, cnt * 100.0 / total_image_cnt))
            frame_id = int(img_p.split('/')[-1].split('.')[0])
            
            if DEBUG:
                print('\n%s' % img_p)
                image = cv2.imread(img_p)

            labelTree = ET.parse(label_p)
            for obj in labelTree.findall('object'):
                cur_cls = obj.find('name').text
                assert cur_cls in classes
                class_id = classes[cur_cls]

                occl = int(obj.find('occluded').text)
                track_id = int(obj.find('trackid').text)
                bbox = obj.find('bndbox')
                bbox = [package_id, video_id, frame_id, track_id,
                        class_id, occl,
                        int(bbox.find('xmin').text),
                        int(bbox.find('ymin').text),
                        int(bbox.find('xmax').text),
                        int(bbox.find('ymax').text)]
                bboxes.append(bbox)

                if DEBUG:
                    if track_id not in track_color:
                        track_color[track_id] = tuple([int(random.random() * 255) for _ in range(3)])
                    color = track_color[track_id]
                    cv2.rectangle(image, tuple(bbox[-4:-2]), tuple(bbox[-2:]), color)
            if DEBUG:
                print('\n%s' % img_p)
                cv2.imshow('image', image)
                cv2.waitKey(1)

            cnt += 1

    # reorder by video_id, then track_id, then frame_id => all labels for a single track are next to each other
    # (matters only if a single image could have multiple tracks)
    bboxes = np.array(bboxes)
    order = np.lexsort((bboxes[:,2], bboxes[:,3], bboxes[:,1]))
    bboxes = bboxes[order,:]
    np.save('labels/' + label_type + '_label.npy', bboxes)

if __name__ == '__main__':
    main('train')
    main('val')

