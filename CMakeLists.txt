# cmake_minimum_required(VERSION 3.0.0)
# project(plan VERSION 0.1.0)


# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# add_executable(plan main.cpp)
# target_link_libraries(plan "${TORCH_LIBRARIES}")
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)
set(CMAKE_PREFIX_PATH "/home/rong/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)