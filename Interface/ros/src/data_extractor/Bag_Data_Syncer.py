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

# class definition

class Solver:
    file_set = set([]) # not thread-save => ok here since constructed in one thread
    
    def __init__(self,file_name):
        assert file_name not in Solver.file_set
        Solver.file_set.add(file_name)
        self.file_name = file_name
        self.file = open(file_name, 'w')

    def record(self):
        raise NotImplementedError

    def stop_record(self):
        self.file.close()
        Solver.file_set.remove(self.file_name)


class Speed_Solver(Solver):
    def __init__(self, file_name):
        super(Speed_Solver, self).__init__(file_name)

    def record(self, data):
        self.file.write('my speed: ' + str(data.sys_time_us) + ' ' + str(data.vehicle_speed) + '\n')

class Radar_Solver(Solver):

    def __init__(self, file_name, radar_config, transfer_type='2D'):
        super(Radar_Solver, self).__init__(file_name)
        assert transfer_type in ('2D')
        
        self.radar_config = radar_config
        self.gen_transfer(transfer_type)
    
    def gen_transfer(self, transfer_type):        
        yaw = self.radar_config['rotation']['yaw']
        self.__sin_yaw = math.sin(yaw)
        self.__cos_yaw = math.cos(yaw)

        self.__offset_x = self.radar_config['translation']['x']
        self.__offset_y = self.radar_config['translation']['y']
        
        # projected_x = x*cos_yaw - y*sin_yaw
        # projected_y = x*sin_yaw + y*cos_yaw
        if transfer_type == '2D':
            self.transfer = lambda x,y: (x*self.__cos_yaw - y*self.__sin_yaw - self.__offset_x, 
                                         x*self.__sin_yaw + y*self.__cos_yaw - self.__offset_y)
    
    def record(self, radar_targets):
        self.file.write(str(len(radar_targets.radar_targets)) + ' ==>>\n')
        for cur_tar in radar_targets.radar_targets:
            x,y = self.transfer(cur_tar.coordinate.x, cur_tar.coordinate.y)
            self.file.write('tar: ' + str(cur_tar.timestamp) + ' ' + str(x) + ',' + str(y) + ' ' + str(cur_tar.velocity.x) + ',' + str(cur_tar.velocity.y) + '\n')

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
                 
# # get bag's meta-data
# bag = rosbag.Bag(bag_path)
# meta_info = bag.get_type_and_topic_info()

# all_msg_types = tuple(meta_info[0].keys())
# all_topics_info = meta_info[1]

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

print('stop record')
