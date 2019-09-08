cmake_minimum_required( VERSION 2.8 )
project( PclMapping )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen
find_package(Eigen3)
include_directories( ${EIGEN3_INCLUDE_DIRS} )

# pcl 
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

#gflags
find_package( gflags )
include_directories( ${gflags_INCLUDE_DIRS} )
add_definitions( ${gflags_DEFINITIONS} )

#PFM
add_library(pfm ${CMAKE_SOURCE_DIR}/3rdParty/pfmLib/ImageIOpfm.cpp)

#elas
#add_subdirectory(${CMAKE_SOURCE_DIR}/3rdParty/alg-ELAS ELAS)
set (LIBELAS_SRC_DIR ${CMAKE_SOURCE_DIR}/3rdParty/libelas)
include_directories(${LIBELAS_SRC_DIR})
FILE(GLOB LIBELAS_SRC_FILES "3rdParty/libelas/*.cpp")
add_library(libelas ${LIBELAS_SRC_FILES} )

add_executable( PclMapping PclMapping.cpp)
target_link_libraries( PclMapping ${OpenCV_LIBS} ${PCL_LIBRARIES} )
target_link_libraries( PclMapping pfm )
target_link_libraries( PclMapping gflags )
target_link_libraries( PclMapping libelas )
