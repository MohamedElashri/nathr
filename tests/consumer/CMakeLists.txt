cmake_minimum_required(VERSION 3.25)
project(nathr_consumer LANGUAGES CXX)

find_package(nathr CONFIG REQUIRED)
add_executable(nathr-consumer main.cpp)
target_link_libraries(nathr-consumer PRIVATE nathr::nathr)
