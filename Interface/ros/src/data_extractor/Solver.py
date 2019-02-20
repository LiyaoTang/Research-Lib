import math

class Solver(object):
    file_set = set([]) # not thread-save => ok here since constructed in one thread
    
    def __init__(self,file_name):
        assert file_name not in Solver.file_set
        Solver.file_set.add(file_name)
        self.file_name = file_name
        self.file = open(file_name, 'w')

    def record(self, data):
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
    
    def record(self, data):
        self.file.write(str(len(data.radar_targets)) + ' ==>>\n')
        for cur_tar in data.radar_targets:
            x, y = self.transfer(cur_tar.coordinate.x, cur_tar.coordinate.y)
            v_x, v_y = self.transfer(cur_tar.velocity.x, 0)
            self.file.write('tar: ' + str(cur_tar.timestamp) + ' ' + str(x) + ',' + str(y) + ' ' + str(v_x) + ',' + str(v_y) + ' ' + str(cur_tar.rcs) + ' ' + str(cur_tar.track_status) + '\n')