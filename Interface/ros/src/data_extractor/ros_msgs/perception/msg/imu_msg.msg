# This is a message to hold data from an IMU (Inertial Measurement Unit)
#
# Accelerations should be in m/s^2 (not in g's), and rotational velocity should be in rad/sec
#
# If the covariance of the measurement is known, it should be filled in (if all you know is the 
# variance of each measurement, e.g. from the datasheet, just put those along the diagonal)
# A covariance matrix of all zeros will be interpreted as "covariance unknown", and to use the
# data a covariance will have to be assumed or gotten from some other source
#
# If you have no estimate for one of the data elements (e.g. your IMU doesn't produce an orientation 
# estimate), please set element 0 of the associated covariance matrix to -1
# If you are interpreting this message, please check for a value of -1 in the first element of each 
# covariance matrix, and disregard the associated estimate.

Header header

uint64 sys_time_us              #/*用于各传感器消息时间同步的时间戳*/

uint64 measurement_time
#unit is us.

uint8 status

# When measurement_span is non-zero, the gyroscope and accelerometer
# measurements are averaged for the period from
# (measurement_time - measurement_span) to measurement_time. Usually,
#      measurement_span = 1 / sampling_frequency.
#
# When measurement_span is 0, angular_velocity and linear_acceleration are
# instantaneous at measurement_time.

float64 measurement_span

float64 linear_acceleration_x
float64 linear_acceleration_y
float64 linear_acceleration_z
float64[9] linear_acceleration_covariance # Row major x, y z

float64 angular_velocity_x
float64 angular_velocity_y
float64 angular_velocity_z
float64[9] angular_velocity_covariance # Row major about x, y, z axes

float64 orientation_pitch
float64 orientation_roll
float64 orientation_yaw

float64 orientation_quternion_1
float64 orientation_quternion_2
float64 orientation_quternion_3
float64 orientation_quternion_4

float64[9] orientation_covariance # Row major about x, y, z axes
