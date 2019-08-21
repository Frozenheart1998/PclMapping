cmake_minimum_required( VERSION 2.8 )
project( PclMapping )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen 
include_directories( "/usr/include/eigen3/" )
find_package(eigen3)
include_directories( ${eigen3_INCLUDE_DIRS} )

# pcl 
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )


#PFM
add_library(pfm ./pfmLib/ImageIOpfm.cpp)

add_executable( PclMapping PclMapping.cpp)
target_link_libraries( PclMapping ${OpenCV_LIBS} ${PCL_LIBRARIES} )
target_link_libraries( PclMapping pfm )
