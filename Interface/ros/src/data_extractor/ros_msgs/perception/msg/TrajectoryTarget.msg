# time information
Header header

TrajectoryTargetPoint[] Trajectory
float64 Dis_Lane2Car # unit: m
float64 Yaw_Lane2Car # unit: rad
float64 target_speed # unit: m/s

float64 theta  # from ato; unit: rad
float64 current_velocity  # from vehicle_speed topic; unit:  m/s
float64 target_velocity  # from ato; unit:  m/s
uint32[] source_ids  # 1. road infos  2. vehicle infos  3. vehicle speed 4,cpt  5,steerAngle
