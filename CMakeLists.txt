cmake_minimum_required(VERSION 2.8)
project( HSVTracker )
find_package( OpenCV REQUIRED )
add_executable( HSVTracker src/main.cpp )
target_link_libraries( HSVTracker ${OpenCV_LIBS} )