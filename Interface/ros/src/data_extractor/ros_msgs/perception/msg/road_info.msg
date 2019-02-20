Header header
uint64 sys_time_us              #/*用于各传感器消息时间同步的时间戳*/
lane_info[] vlanes              #/*存储车道信息*/
lane_mark_info[] vlane_marks    #/*存储车道线信息*/
float64 lane_departure_angle     #/*!< 道路与车身坐标系的夹角(弧度单位[-PI,PI]) */
float64 lane_departure_distance  # /*!< 自车与车道中心线的距离 */