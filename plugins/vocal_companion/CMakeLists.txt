cmake_minimum_required(VERSION 3.15)
project(vocal_companion_plugin)

if(MODC_BUILD_PLUGIN)
  add_subdirectory(../modcompose/juce juce)
  juce_add_plugin(VocalCompanion
    COMPANY_NAME "ModCompose"
    PLUGIN_NAME  "VocalCompanion"
    FORMATS      VST3
    SOURCES
      PluginProcessor.cpp
      PluginProcessor.h
      PluginEditor.cpp
      PluginEditor.h
  )
  target_link_libraries(VocalCompanion PRIVATE pybind11::embed)
else()
  pybind11_add_module(vocal_companion_plugin bindings/plugin.cpp)
  set_target_properties(vocal_companion_plugin PROPERTIES PREFIX "" OUTPUT_NAME "vocal_companion_plugin")
endif()
