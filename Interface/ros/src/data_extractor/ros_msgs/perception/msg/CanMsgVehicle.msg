# time information
Header header

# can message
float64 Vx   #/*  Vehicle velocity,  单位m/s  */
int16 Vx_update # /* 1:updated,   0: non-updated */
float64 V_whl_FL # /*  左前轮轮速  单位m/s */
int16 V_whl_FL_update # /* 1:updated,   0: non-updated */
float64 V_whl_FR   # /*  右前轮轮速  单位m/s */
int16 V_whl_FR_update # /* 1:updated,   0: non-updated */
float64 V_whl_RL    # /*  左后轮轮速  单位m/s */
int16 V_whl_RL_update # /* 1:updated,   0: non-updated */
float64 V_whl_RR # /*  右后轮轮速  单位m/s */
int16 V_whl_RR_update # /* 1:updated,   0: non-updated */

float64 steerAngle   #/* 方向盘转角，单位deg */
int16 steerAngle_update # /* 1:updated,   0: non-updated */

float64 Ax              # /* 车身纵向加速度，单位m/s2 */
int16 Ax_update        # /* 1:updated,   0: non-updated */
float64 Ay             # /* 车身横向加速度，单位m/s2 */
int16 Ay_update        # /* 1:updated,   0: non-updated */
float64 yawrate        # /*车身航偏角速度，单位rad/s */
int16 yawrate_update   # /* 1:updated,   0: non-updated */

