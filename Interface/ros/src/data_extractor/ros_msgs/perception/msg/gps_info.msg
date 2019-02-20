# @file gps_info.msg

#如果浮点数据字段无效表示为NaN
Header header
uint64  fix_type        # gps固定类型 0: no fix, 1: dead reckoning only, 2: 2D-fix, 3: 3D-fix, 4: GNSS + dead reckoning combined, 5: time only fix
uint64  utc_time_us     # UTC时间  (单位 us)
uint64  sys_time_us     # 系统时间 (单位 us)
float64 lon             # 经度 (单位 deg)
float64 lat             # 纬度 (单位 deg)
float64 height          # 高层 Height above ellipsoid      (单位 m)
float64 height_msl      # 高层 Height above mean sea level (单位 m)
float64 vel_2d          # 地面2D速度  (单位 m/s)
float64 vel_n           # ned速度n分量(单位 m/s)
float64 vel_e           # ned速度e分量(单位 m/s)
float64 vel_d           # ned速度d分量(单位 m/s)
float64 head_mot        # 方向 Heading of motion (2-D)  (单位 rad (-pi, pi])
float64 head_veh        # 方向 Heading of vehicle (2-D) (单位 rad (-pi, pi])
float64 acc_ver         # 水平定位精度 (单位 m)
float64 acc_hor         # 垂直定位精度 (单位 m)
float64 acc_speed       # 速度精度     (单位 m/s)
float64 acc_head        # 方向精度 Heading accuracy estimate (both motion and vehicle) (单位 rad)

#确定还需要那些字段 
#字段有效性如何表示
#确定字段的类型
#字段的数据区间