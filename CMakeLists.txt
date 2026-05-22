cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(NeuralNetwork
    Main.cpp
    include/NeuralNetwork.hpp
)

target_include_directories(NeuralNetwork PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
