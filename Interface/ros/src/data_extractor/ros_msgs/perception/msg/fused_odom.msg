Header header
uint64 time_ms
uint64 time_us
uint64 time_ns
float64 time_s_recall_dt   # delta time recall_time - called_time in second(s)
float64 longitude   # odom longitude
float64 latitude    # odom latitude

#  y(north)
#  ^
#  |
#  | 
#  o------>x(east)
float64 mercator_x  # odom mercator_x
float64 mercator_y  # odom mercator_y
float64 mercator_z  # odom mercator_z
float64 acc_x_filtered
float64 acc_y_filtered
float64 acc_z_filtered
float64 gyro_x_filtered
float64 gyro_y_filtered
float64 gyro_z_filtered
float64 vel_x
float64 vel_y
float64 vel_z
float64 roll
float64 pitch
float64 yaw
float64 gyro_bias_x
float64 gyro_bias_y
float64 gyro_bias_z
float64 acc_scale_x
float64 acc_scale_y
float64 acc_scale_z
float64 acc_bias_x
float64 acc_bias_y
float64 acc_bias_z
# quaternion
float64 q1
float64 q2
float64 q3
float64 q4
float64 yaw_rad     # rad/s
float64 speed       # m/s
float64 current_day_sec # current_day_second
int64 px2_time_us   # 

# gps
float64 gps_longitude
float64 gps_latitude
float64 gps_heading
float64 gps_speed
uint64  gps_utc_time_us

int64 section_id    # section_id in had map, if not valid, return -1
int64 lane_id       # lane id in had map, if not valid, return -1

bool  odom_healthy  # indicate odom health status

# guide_point, last localization result
perception_msgs/guide_point guide_point
