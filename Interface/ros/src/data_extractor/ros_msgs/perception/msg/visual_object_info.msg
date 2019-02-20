int32 parts
float64[] pts8                   #//2Dto3D, pts8.resize(16), x, y...
float64[] front_upper_left         #//front box upper-left corner: x1, y1
float64[] front_lower_right        #//front box  lower-right corner: x2, y2
float64[] back_upper_left            #//front box upper-left corner: x1, y1
float64[] back_lower_right           #//front box  lower-right corner: x2, y2
int32 id = 0                         #// object id per frame
int32 turn_signal_light
int32 brake_light
int32 type                           #// Object classification type.
float64[] type_probs                 #// Probability of each type, used for track type.
float64[] object_feature
int32 internal_type                  #// Internal object classification type.
float64[] internal_type_probs   #// Internal probability of each type, used for track type.

#// 2D bounding box information (in pixel)
#// upper-left corner: x1, y1
float64[] upper_left
#// lower-right corner: x2, y2
float64[] lower_right
#// tmp for upper-left & lower-right corner point
float64[] upper_left_tmp
float64[] lower_right_tmp

#// this is `alpha` angle from KITTI: Observation angle of object, ranging [-pi..pi]
float64 alpha
float64 score
#// Truncation ratio prediction from DNN. 0.0 means non-truncated box ( or < threshold, ex: 0.05)
float64 truncated_horizontal
float64 truncated_vertical

#// 3D bounding box information
#// main direction
float64[] direction

#// the yaw angle, theta = 0.0 <=> direction = (1, 0, 0)
#// `theta` is `rotation_y` from KITTI: Rotation ry around Y-axis
#// in camera coordinates before track [-pi..pi]
#// in world coordinates in and after track [-pi..pi]
float64 theta

#// GROUND center of the object (cx, cy, z_min)
#// From the 2409 test data, it's just the CENTER of the object (cx, cy, cz)
#// in camera coordinates before track [-pi..pi]
#// in world coordinates in and after track [-pi..pi]
float64[] center

float64[] position_uncertainty#// noise covariance matrix for uncertainty of position and velocity
float64[] velocity_uncertainty

#// size of the oriented bbox, length is the size in the main direction
float64 length
float64 width
float64 height

float64 distance    #// Distance to object physical center from lidar origin
int32 distance_status

float64 azimuth     #// Azimuth angle viewed in lidar space (radian)

int32 track_id      #// tracking information
float64 track_score
float64[] velocity

float64 tracking_time       #// age of the tracked object
float64 latest_tracked_time
