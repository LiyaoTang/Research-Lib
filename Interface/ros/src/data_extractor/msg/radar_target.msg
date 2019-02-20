Header header
uint64 timestamp  # unit: us
uint32 group_id  # radar id (front: 0  left front: 1  right front:2
 # back: 3  left back: 4  right back: 5)
uint32 radar_type # detail radar version
 # DELPHI_ESR = 0, DELPHI_RSDS = 1,
 # CONTI_408 = 2, CONTI_208 = 3,
 # append if more types will be supported
 # unknown = other values
uint32 target_id  # target id given by radar itself
uint32 track_status  # target status (need to merge with conti408 obj_MeasState)
 # 0x0:No_Target
 # 0x1:New_Target
 # 0x2:New_Updated_Target
 # 0x3:Updated_Target
 # 0x4:Coasted_Target
 # 0x5:Merged_Target
 # 0x6:Invalid_Coasted_Target
 # 0x7:New_Coasted_Target
uint32 moving_status  # 0x0: moving
 # 0x1: stationary
 # 0x2: oncoming
 # 0x3: stationary candidate
 # 0x4: unknown
 # 0x5: crossing stationary
 # 0x6: crossing moving
 # 0x7: stopped
uint32 obj_type  # 0x0: point
 # 0x1: car
 # 0x2: truck
 # 0x3: pedestrian
 # 0x4: motorcycle
 # 0x5: bicycle
 # 0x6: wide
 # 0x7: reserved
 # 0x8: bridge
 # 0x9: curb
coord3d coordinate  # relative position (unit: m  up, front, left is positive)
coord3d velocity  # relative velocity(unit: m/s  awaying is positive)
bool is_radial_velocity  # set true if it's radial velocity
coord3d acceleration  # relative acceleration (unit: m/s^2  awaying is positive)
coord3d coordinate_rms  # standard deviation of position(unit: m)
coord3d velocity_rms  # standard deviation of velocity(unit: m/s)
coord3d acceleration_rms  # standard deviation of acceleration(unit: m/s^2)
coord3d size_of_box  # length, width, height (unit: m) default: -1
float64 rcs  # Radar Cross Section(unit: dBm^2)
float64 orientation_angle  # Orientation angle(unit: rad)
float64 orientation_angle_rms  # standard deviation of orientation angle(unit: rad)
float64 prob_of_exist  # probability of existence(use upper limit)  default: -1
float64 prob_of_objtype  #probability of object type.
float64 lifetime  # the current lifetime of track(unit: s)  default: -1
bool is_acceleration_valid # set true if acceleration is valid
bool is_coordinate_rms_valid # set true if valid
bool is_velocity_rms_valid # set true if valid
bool is_acceleration_rms_valid # set true if valid
bool is_orientation_angle_valid # set true if valid
bool is_orientation_angle_rms_valid # set true if valid
