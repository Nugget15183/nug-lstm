cmake_minimum_required(VERSION 4.0)
project(lstm)

set(CMAKE_CXX_STANDARD 20)

add_executable(lstm src/main.cpp
        src/LSTM.cpp
        src/LSTM.h)
