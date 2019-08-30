cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
project(model)

file(GLOB_RECURSE _SRCS "src/*.[hc]pp") # serch for files
message("found files in src = ${_SRCS}")
message("project src dir = ${PROJECT_SOURCE_DIR}")

# generate .so & include local files
add_library(${PROJECT_NAME} SHARED ${_SRCS})

# opencv
set("OpenCV_DIR" "/usr/local/include/opencv4")
find_package(OpenCV REQUIRED)
message(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
message(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# libtorch
set("Torch_Dir" "/usr/local/include/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${Torch_Dir})
include_directories(${Torch_Dir}/include) # include the header files (as libtorch not installed into system)
find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

# target to compile
add_executable(cpp_model_test src/unit_test/unit_test.cpp)
target_link_libraries(cpp_model_test ${PROJECT_NAME}) # link
# add_dependencies(cpp_model_test ${PROJECT_NAME})