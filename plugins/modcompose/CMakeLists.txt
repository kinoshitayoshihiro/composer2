cmake_minimum_required(VERSION 3.15)
project(modcompose_plugin)

if(MODC_BUILD_PLUGIN)
  add_subdirectory(juce)
  juce_add_plugin(ModCompose
    COMPANY_NAME "ModCompose"
    PLUGIN_NAME  "ModCompose"
    FORMATS      VST3 CLAP
    SOURCES
      PluginProcessor.cpp
      PluginProcessor.h
      PluginEditor.cpp
      PluginEditor.h
      plugin.cpp
      ringbuffer.h
  )
  target_link_libraries(ModCompose PRIVATE pybind11::embed)
else()
  pybind11_add_module(modcompose_plugin plugin.cpp)
  set_target_properties(modcompose_plugin PROPERTIES PREFIX "" OUTPUT_NAME "modcompose_plugin")
endif()
