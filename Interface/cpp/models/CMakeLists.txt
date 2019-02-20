file(GLOB _SRCS "*.[hc]xx") # serch for files

# INTERFACE for header files only
add_library(${PROJECT_NAME} INTERFACE ${_SRCS})

# make sure the compiler can find include files for our library
# when other libraries or executables link to this library
target_include_directories (${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})