cmake_minimum_required(VERSION 3.0.0)

project(PicoPebble VERSION 0.1.0)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/eigen
)

include_directories(${MPI_INCLUDE_PATH})

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/Common
    ${PROJECT_SOURCE_DIR}/include/Activations
    ${PROJECT_SOURCE_DIR}/include/DataLoader
    ${PROJECT_SOURCE_DIR}/include/Layers
    ${PROJECT_SOURCE_DIR}/include/Losses
    ${PROJECT_SOURCE_DIR}/include/Metrics
    ${PROJECT_SOURCE_DIR}/include/Module
    ${PROJECT_SOURCE_DIR}/include/Sequential
    ${PROJECT_SOURCE_DIR}/include/Trainers
    ${PROJECT_SOURCE_DIR}/src/Trainers
)


# source files
file(GLOB SOURCES "src/*/*/*.cpp" "src/*/*.cpp" examples/dnn.cpp)

find_package(Threads REQUIRED)

message(STATUS "Find Threads: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "Find MPI: ${MPI_LIBRARIES}")

add_executable(example ${SOURCES})

# target_link_libraries(example ${MPI_LIBRARIES} Threads::Threads)
target_link_libraries(example ${MPI_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
