cmake_minimum_required (VERSION 3.0.0)

project(TinyDistributedDeepLearning VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/eigen
)

include_directories(${MPI_INCLUDE_PATH})

include_directories(
    ${PROJECT_SOURCE_DIR}/include
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
file(GLOB SOURCES "src/*/*.cpp" examples/dnn.cpp)


add_executable(example ${SOURCES})
target_link_libraries(example ${MPI_LIBRARIES})
