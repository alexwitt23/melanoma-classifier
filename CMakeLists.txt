project(main-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread ${TORCH_CXX_FLAGS}")

add_executable(main-app src/main.cpp)
target_link_libraries(main-app "${TORCH_LIBRARIES}")
set_property(TARGET main-app PROPERTY CXX_STANDARD 14)