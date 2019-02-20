# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 2.8.3)
project(immediate_msgs)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg message_generation)

## Declare ROS messages and services
add_message_files(FILES SuspectedType.msg
SuspectedPoint.msg
ImmediateTrajectory.msg 
ImmediateTrajectoryPoint.msg
)
#add_service_files(FILES AddTwoInts.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES  std_msgs)

## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

#add_executable(trajectory src/trajectory.cpp )
#target_link_libraries(trajectory ${catkin_LIBRARIES})
#add_dependencies(trajectory immediate_trajectory_module_generate_messages_cpp)
# %EndTag(FULLTEXT)%
