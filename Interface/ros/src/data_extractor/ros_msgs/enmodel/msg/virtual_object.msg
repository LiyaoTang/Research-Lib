Header header
int32 obj_id            #/*!< 目标编号 */
uint8 type              #/*!< 类型: perception ObjectType */

float64 x_v             #/*!< 纵向距离(米) */
float64 heading_c       #/*!< 中心点heading角(弧度单位) */
float64 heading_l       #/*!< 左侧端点heading角, 相机(弧度单位) */
float64 heading_r       #/*!< 右侧端点heading角, 相机(弧度单位) */
float64 heading_speed   #/*!< 运行的速度方向(弧度单位) */
float64 speed           #/*!< 速度, 雷达(米/秒) */

##/*! 注意：当只有雷达检测出来的时候都设置为100米，宽度为-100米*/
float64 height          #/* !< object 高度(米) */
float64 width           #/* !< object 宽度(米) */
float64 length          #/* !< object 长度(米) */
bool is_top_surface     #/* !< 是否为车辆的上表面，相对x轴方向 */

float64 prob            #/* !< object 置信度0-1 */
