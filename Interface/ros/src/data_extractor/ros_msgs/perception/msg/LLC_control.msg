# time information
Header header

float64 LLC_WhlAng_Stp_Deg # 方向盘转角请求，单位degree，左转为正，右转为负。
float64 LLC_WhlAng_Spd_Req # 方向盘转速请求，一般为100。
float64 LLC_AxStpStatus #  制动请求类型：
			# 0: No Action
			# 1: Apply by Ax_setp
			# 2: Apply by pressure
			# 3: Park Brake
			# 4: Precharge
			# 5: Full Brake

float64 LLC_Ax_StpBrake_ms2 # 制动减速度请求，单位m/s2。只有当LLC_AxStpStatus=1时，该值有效。
float64 LLC_Pres_StpBrake_Bar # 制动压力请求，单位bar。只有当LLC_AxStpStatus=2时，该值有效。
float64 LLC_ThrottletSet # 油门开度请求，0-1。1表示全油门驱动
float64 LLC_GearControlAuto # 换挡请求
			    # -9 = position P (corresponds to gear number -9)
			    # -1 = position R (corresponds to gear number -1...-8)
			    # 0 = position N (corresponds to gear number =0)
			    # 1 = position D (corresponds to gear number >0)
			    # 2 = position M (manual gear selection / Manumatic)

uint32[] source_ids  # 1. road infos  2. vehicle infos  3. vehicle speed 4,cpt  5,steerAngle 
			# 6,Inner_monitor
