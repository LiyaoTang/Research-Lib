# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 2.8.3)
project(decision_make_msg)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg enmodel_msgs perception_msgs hdmap_interface_msgs message_generation)

## Declare ROS messages and services
add_message_files(FILES decisionmake.msg 
decisionmake_pack.msg
hmiDecision.msg
target_object_info.msg
lanepoint.msg
lanes.msg
lanetopo.msg
)
#add_service_files(FILES AddTwoInts.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES  perception_msgs hdmap_interface_msgs std_msgs enmodel_msgs)

## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

#add_executable(talker1 src/talker.cpp )
#target_link_libraries(talker1 ${catkin_LIBRARIES})
#add_dependencies(talker1 decision_make_module_generate_messages_cpp)
# %EndTag(FULLTEXT)%
