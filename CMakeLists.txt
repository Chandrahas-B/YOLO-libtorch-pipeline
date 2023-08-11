cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ObjectDetection)

set(CMAKE_PREFIX_PATH ~/cppLib/opencv) ## add your opencv path
set(CMAKE_PREFIX_PATH ~/cppLib/vision) ## change the torchvision path
set(CMAKE_PREFIX_PATH ~/cppLib/libtorchGPU) ## change the libtorch path (with or without GPU)
set(CMAKE_PREFIX_PATH ~/cppLib/cnpy) ## add the numcpp path


find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TorchVision REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ObjectDetection src/main.cpp)
include(FetchContent)
FetchContent_Declare(NumCpp
        GIT_REPOSITORY https://github.com/dpilger26/NumCpp
        GIT_TAG Version_2.9.0)
FetchContent_MakeAvailable(NumCpp)

target_link_libraries(${PROJECT_NAME}
    NumCpp::NumCpp
)

# add_executable(ObjectDetection src/main.cpp src/model.h src/dataloader.h)
# add_executable(ObjectDetection src/train.cpp src/model.h src/dataloader.h)
add_executable(ObjectDetection src/inference.cpp)
target_link_libraries(ObjectDetection "${TORCH_LIBRARIES}")
target_link_libraries(ObjectDetection TorchVision::TorchVision)
target_link_libraries(ObjectDetection "${OpenCV_LIBS}" )
set_property(TARGET ObjectDetection PROPERTY CXX_STANDARD 17)
