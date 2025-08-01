cmake_minimum_required(VERSION 3.14)
project(compressai_compressai_speedup_codec)

message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message(STATUS "You Can Set GITHUB_PROXY_PREFIX as github proxy url prefix, e.g. GitHub proxy URL prefix, e.g. https://ghfast.top/")

set(GITHUB_PROXY_PREFIX "" CACHE STRING "GitHub proxy URL prefix, e.g. https://ghfast.top/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")  # 安装后可执行文件在 bin/，lib 在 ../lib

# 输出目录配置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(ExternalProject)

# 添加 opencv 构建任务
ExternalProject_Add(
    opencv_ext
    URL ${GITHUB_PROXY_PREFIX}https://github.com/opencv/opencv/archive/refs/tags/4.9.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    PREFIX ${CMAKE_BINARY_DIR}/_deps
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/third_party_install/opencv
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_LIST=core,imgcodecs,imgproc
        -DBUILD_IPP_IW=OFF
        -DWITH_IPP=OFF
        -DBUILD_opencv_videoio=OFF
        -DWITH_FFMPEG=OFF
        -DWITH_GSTREAMER=OFF
        -DWITH_MSMF=OFF
        -DWITH_V4L=OFF
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_TESTS=OFF
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_EXAMPLES=OFF
)

# 添加 cnpy 构建任务
ExternalProject_Add(
    cnpy_ext
    GIT_REPOSITORY ${GITHUB_PROXY_PREFIX}https://github.com/rogersce/cnpy.git
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    PREFIX ${CMAKE_BINARY_DIR}/_deps
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/third_party_install/cnpy
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 
)

# 添加 onnxruntime 构建任务
ExternalProject_Add(
    onnxruntime_ext
    URL ${GITHUB_PROXY_PREFIX}https://github.com/microsoft/onnxruntime/releases/download/v1.15.0/onnxruntime-linux-x64-1.15.0.tgz
    PREFIX ${CMAKE_BINARY_DIR}/_deps
    CONFIGURE_COMMAND ""   # 跳过配置
    BUILD_COMMAND ""       # 跳过构建
    INSTALL_COMMAND ""     # 跳过安装
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# 添加 xtl 构建任务
ExternalProject_Add(
    xtl_ext
    URL ${GITHUB_PROXY_PREFIX}https://github.com/xtensor-stack/xtl/archive/refs/tags/0.8.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    PREFIX ${CMAKE_BINARY_DIR}/_deps
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/third_party_install/xtl
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
)

# 添加 xtensor 构建任务
ExternalProject_Add(
    xtensor_ext
    URL ${GITHUB_PROXY_PREFIX}https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.26.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    PREFIX ${CMAKE_BINARY_DIR}/_deps
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/third_party_install/xtensor
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/third_party_install/xtl/share/cmake/xtl
)

# 等待所有外部项目构建完成
add_dependencies(xtensor_ext xtl_ext)

# 包含头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party/ryg_rans
    ${CMAKE_BINARY_DIR}/third_party_install/opencv/include/opencv4
    ${CMAKE_BINARY_DIR}/third_party_install/cnpy/include
    ${CMAKE_BINARY_DIR}/_deps/src/onnxruntime_ext/include
    ${CMAKE_BINARY_DIR}/third_party_install/xtensor/include
    ${CMAKE_BINARY_DIR}/third_party_install/xtl/include
)

# 查找 src 下所有 cpp 文件
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# 创建动态库
add_library(cmpai_shared SHARED ${SRC_FILES})
set_target_properties(cmpai_shared PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    OUTPUT_NAME cmpai
)

# 创建静态库
add_library(cmpai_static STATIC ${SRC_FILES})
set_target_properties(cmpai_static PROPERTIES
    OUTPUT_NAME cmpai
)

# 设置依赖关系
add_dependencies(cmpai_shared opencv_ext cnpy_ext onnxruntime_ext xtensor_ext xtl_ext)
add_dependencies(cmpai_static opencv_ext cnpy_ext onnxruntime_ext xtensor_ext xtl_ext)

# 链接库
target_link_libraries(cmpai_shared
    ${CMAKE_BINARY_DIR}/third_party_install/opencv/lib64/libopencv_core.so
    ${CMAKE_BINARY_DIR}/third_party_install/opencv/lib64/libopencv_imgcodecs.so
    ${CMAKE_BINARY_DIR}/third_party_install/opencv/lib64/libopencv_imgproc.so
    ${CMAKE_BINARY_DIR}/third_party_install/cnpy/lib/libcnpy.so
    ${CMAKE_BINARY_DIR}/_deps/src/onnxruntime_ext/lib/libonnxruntime.so
)

target_link_libraries(cmpai_static
    ${CMAKE_BINARY_DIR}/third_party_install/opencv/lib64/libopencv_core.so
    ${CMAKE_BINARY_DIR}/third_party_install/opencv/lib64/libopencv_imgcodecs.so
    ${CMAKE_BINARY_DIR}/third_party_install/opencv/lib64/libopencv_imgproc.so
    ${CMAKE_BINARY_DIR}/third_party_install/cnpy/lib/libcnpy.so
    ${CMAKE_BINARY_DIR}/_deps/src/onnxruntime_ext/lib/libonnxruntime.so
)

# 编译选项
target_compile_options(cmpai_shared PRIVATE 
    -Wall 
    -Wextra 
    -Wno-unused-parameter
    -O3
    -march=native
    -std=c++17
    -D_GLIBCXX_USE_CXX11_ABI=1
)

target_compile_options(cmpai_static PRIVATE 
    -Wall 
    -Wextra 
    -Wno-unused-parameter
    -O3
    -march=native
    -std=c++17
    -D_GLIBCXX_USE_CXX11_ABI=1
)

# 编译定义
target_compile_definitions(cmpai_shared PRIVATE
    _GLIBCXX_USE_CXX11_ABI=1
)

target_compile_definitions(cmpai_static PRIVATE
    _GLIBCXX_USE_CXX11_ABI=1
)

# 创建可执行文件
add_executable(cmpai-cli ${PROJECT_SOURCE_DIR}/src/cli.cpp)
target_link_libraries(cmpai-cli cmpai_shared)
add_dependencies(cmpai-cli cmpai_shared)

# 安装规则
install(TARGETS cmpai_shared cmpai_static cmpai-cli
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/third_party_install/opencv/lib64/
    DESTINATION lib
    FILES_MATCHING PATTERN "*.so*")

install(DIRECTORY ${CMAKE_BINARY_DIR}/third_party_install/cnpy/lib/
    DESTINATION lib
    FILES_MATCHING PATTERN "*.so*")

install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/src/onnxruntime_ext/lib/
    DESTINATION lib
    FILES_MATCHING PATTERN "*.so*")

# 安装模型文件到bin目录
install(DIRECTORY ${PROJECT_SOURCE_DIR}/models/
    DESTINATION bin/models
    FILES_MATCHING PATTERN "*.onnx" PATTERN "*.npz"
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 可选：创建 pkg-config 文件（如果不需要可以注释掉这部分）
configure_file(
    "${PROJECT_SOURCE_DIR}/cmpai.pc.in"
    "${CMAKE_BINARY_DIR}/cmpai.pc"
    @ONLY
)
install(FILES "${CMAKE_BINARY_DIR}/cmpai.pc"
    DESTINATION lib/pkgconfig
)


add_custom_target(default_all ALL
    DEPENDS cmpai_shared cmpai_static
)