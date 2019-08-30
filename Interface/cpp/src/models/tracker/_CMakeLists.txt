cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tracker_torch)

find_package(Torch REQUIRED)
# message("============================${TORCH_LIBRARIES}") # print out linked lib

# add_executable(tracker_torch tracker_torch.cpp)
target_link_libraries(tracker_torch "${TORCH_LIBRARIES}")
set_property(TARGET tracker_torch PROPERTY CXX_STANDARD 11)
