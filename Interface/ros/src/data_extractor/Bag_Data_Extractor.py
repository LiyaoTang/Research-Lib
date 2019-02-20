#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append('../../devel/lib/python3/dist-packages/')
if sys.getdefaultencoding() != 'utf-8':
    print('forcing default to utf-8')
    reload(sys)
    sys.setdefaultencoding('utf-8')

import math
import rospy
import rosbag
import threading # not used
import perception_msgs.msg as percept_msg
from collections import namedtuple
from optparse import OptionParser
from Solver import *

# parsing args
parser = OptionParser()
parser.add_option('-f', '--file', dest='bag_file', default='[20180630保定][15s][连续换道].bag', help='bag file name', metavar='FILE')
parser.add_option('-d', '--dir', dest='bag_dir', default='../../../../L3-Apollo-Data/', help='bag file dir', metavar='FILE')
parser.add_option('-s', '--save_dir', dest='save_dir', default='./', help='dir to save output', metavar='FILE')
(options, args) = parser.parse_args()

bag_file = options.bag_file
bag_dir = options.bag_dir.rstrip('/')+'/'
save_dir = options.save_dir.rstrip('/')+'/'

bag_path = bag_dir + bag_file
save_path = save_dir + bag_file

# default configuration
radar_types = ('fl', 'fr', 'bl', 'br')
# roatation: counter-clockwise; translation: front +x, left +y
radar_configs = {'fl': {'rotation': {'pitch': 0, 'yaw': 1.2422, 'roll': 0}, 'translation': {'x': 0.6, 'y': 0.96, 'z': 0}},
                 'fr': {'rotation': {'pitch': 0, 'yaw': -1.211, 'roll': 0}, 'translation': {'x': 0.6, 'y': -0.95, 'z': 0}},
                 'bl': {'rotation': {'pitch': 0, 'yaw': 1.2577174, 'roll': 0}, 'translation': {'x': -3.62, 'y': 0.92, 'z': 0}},
                 'br': {'rotation': {'pitch': 0, 'yaw': -1.91468, 'roll': 0}, 'translation': {'x': -3.59, 'y': -0.96, 'z': 0}}}

# configuring ros node
focused_topics = {'radar': {'fl': '/radar/left_front_targets',
                            'fr': '/radar/right_front_targets',
                            'bl': '/radar/left_back_targets',
                            'br': '/radar/right_back_targets'},
                  'speed': ('/vehicle_speed')}                  
data_types = {'radar': percept_msg.radar_targets,
              'speed': percept_msg.vehicle_speed}
             
# instantiate solver
speed_solver = Speed_Solver(save_path + '_speed')

radar_solvers = {}
for cur_type in radar_types:
    file_name = save_path + '_' + cur_type + '_radar'
    radar_solvers[cur_type] = (Radar_Solver(file_name, radar_configs[cur_type]))

# start ros node for subscribing
rospy.init_node('listener', anonymous=True)

# link solver to topic
rospy.Subscriber(focused_topics['speed'], data_types['speed'], speed_solver.record)

for cur_radar_type in focused_topics['radar'].keys():
    rospy.Subscriber(focused_topics['radar'][cur_radar_type], data_types['radar'], radar_solvers[cur_radar_type].record)

# recording
rospy.spin()

speed_solver.stop_record()
for solver in radar_solvers.values():
    solver.stop_record()

print('stop record & start synchronizing')

# sync_base = 
