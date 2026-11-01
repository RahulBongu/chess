cmake_minimum_required(VERSION 3.22.1)
project(ChalEngine)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(chal-engine SHARED
  src/main/cpp/chal.c
  src/main/cpp/chal_bridge.cpp
)

target_compile_definitions(chal-engine PRIVATE
  main=chal_cli_main
  printf=chal_bridge_printf
  fflush=chal_bridge_fflush
)

target_include_directories(chal-engine PRIVATE
  src/main/cpp
)

target_link_libraries(chal-engine PRIVATE
  m
  log
)
