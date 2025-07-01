cmake_minimum_required(VERSION 3.15)
project(modcompose_root)

option(MODC_BUILD_PLUGIN "Build plugin" OFF)

if(MODC_BUILD_PLUGIN)
    add_subdirectory(plugins/modcompose)
endif()
