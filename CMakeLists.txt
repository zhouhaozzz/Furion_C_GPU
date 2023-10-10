# 本CMakeLists.txt的project名称
# 会自动创建两个变量，PROJECT_SOURCE_DIR和PROJECT_NAME
# ${PROJECT_SOURCE_DIR}：本CMakeLists.txt所在的文件夹路径
# ${PROJECT_NAME}：本CMakeLists.txt的project名称
cmake_minimum_required(VERSION 3.26)
project(Furion LANGUAGES CUDA CXX)
#set(Furion "CMake_Project")

set(CMAKE_CXX_STANDARD 17)
#set(CUDA)


#include_directories(C:/software/C_library/eigen-3.4.0)
#include_directories(C:/software/C_library/MPI/Include)
include_directories(${CMAKE_SOURCE_DIR})

#find_package ( MPI )

#message ( STATUS  "MPI_FOUND=${MPI_FOUND}" )
#message ( STATUS  "MPI_CXX_INCLUDE_DIRS=${MPI_CXX_INCLUDE_DIRS}" )
#message ( STATUS  "MPI_LIBRARIES=${MPI_LIBRARIES}" )
#if ( MPI_FOUND )
#	list ( APPEND PRJ_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS} )
#	list ( APPEND PRJ_LIBRARIES ${MPI_LIBRARIES} )
#endif ()


# 定义头文件路径对应的名字
#通用与辅助
set(DISPLAY_H     "${CMAKE_SOURCE_DIR}/src/display")
set(FUR_SCRI_H     "${CMAKE_SOURCE_DIR}/src/furion_scri")
set(IMP_DIS_SRW_H     "${CMAKE_SOURCE_DIR}/src/import_and_display_srw")
set(OPT_ELE_H     "${CMAKE_SOURCE_DIR}/src/optical_element")
set(PROPAGATION_H     "${CMAKE_SOURCE_DIR}/src/propagation")
set(SURFACE_H     "${CMAKE_SOURCE_DIR}/src/surface")
set(XOP_H     "${CMAKE_SOURCE_DIR}/src/xop")
set(SRC_H     "${CMAKE_SOURCE_DIR}/src")
set(CUDA_H "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include")

add_subdirectory(${CMAKE_SOURCE_DIR}/src)