cmake_minimum_required(VERSION 3.14)
project(wavelet_codec CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Download stb_image.h (header-only image loader)
set(STB_DIR "${CMAKE_CURRENT_BINARY_DIR}/stb")
file(MAKE_DIRECTORY "${STB_DIR}")

if(NOT EXISTS "${STB_DIR}/stb_image.h")
    message(STATUS "Downloading stb_image.h ...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
        "${STB_DIR}/stb_image.h"
        STATUS dl_status)
    list(GET dl_status 0 dl_code)
    if(NOT dl_code EQUAL 0)
        message(FATAL_ERROR "Failed to download stb_image.h")
    endif()
endif()

add_executable(benchmark main.cpp)
target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${STB_DIR}
)

# Optimisation flags
if(MSVC)
    target_compile_options(benchmark PRIVATE /O2 /EHsc)
else()
    target_compile_options(benchmark PRIVATE -O2)
endif()
