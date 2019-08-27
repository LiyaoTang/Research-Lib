cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_model)

find_package(Torch REQUIRED)
message("============================${TORCH_LIBRARIES}")

add_executable(torch_model torch_model.cpp)
target_link_libraries(torch_model "${TORCH_LIBRARIES}")
set_property(TARGET torch_model PROPERTY CXX_STANDARD 11)
