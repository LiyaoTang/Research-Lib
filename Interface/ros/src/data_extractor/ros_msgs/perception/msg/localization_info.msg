Header header
float32 timestamp
uint64 time_ms
uint64 time_us
uint64 time_ns
uint64 sys_time_us
float64 time_s_recall_dt   # delta time recall_time - called_time in second(s)
float64 lon
float64 lat
float32 heading_absolute
float32 heading_relative
float32 heading_had
int16 pos
int32 is_updated
int32 initialized
# 0：　质量好
# 1: 
# 2:
# 3:??
int32 level 
float64 quality

int64 section_id
int64 lane_id

# vehicle_speed
float64 vehicle_speed

perception_msgs/road_info road_info

# odom_lon
float64 odom_longitude
float64 odom_latitude
float64 odom_yaw_rad

# just for display curve
float64[] display

# guide_point
perception_msgs/guide_point guide_point
