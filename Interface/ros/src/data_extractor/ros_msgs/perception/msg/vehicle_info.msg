Header header
float64 heading_lane   #/*!< 当前车道与Host车辆坐标系x轴夹角,绕x轴旋转，弧度单位,取值范围：(0 – 2π) */
float64 dist_to_line   #/*!< 到右侧车道线距离 */
bool is_pressed_line   #/*!< Host车是否压到车道线上 */
int32 pressed_lane_id  #/*!< Host车压到的车道ID */
float64 current_speed  #/*!< 当前车速 */
float64 longitude      #/*!< 高精地图坐标 x 经度 */
float64 latitude       #/*!< 高精地图坐标 y 纬度 */
float64 height         #/*!< 高精地图坐标 z 高程 */

#/*!< 北东天坐标系 高精地图坐标到车身坐标的变换角*/
float64 roll           #/*!< 参考INS/IMU输出,弧度单位,取值范围：(0 – 2π) */
float64 pitch          #/*!< 参考INS/IMU输出,弧度单位, 取值范围：(0 – 2π)*/
float64 yaw            #/*!< 参考INS/IMU输出,弧度单位,取值范围：(0 – 2π) */

