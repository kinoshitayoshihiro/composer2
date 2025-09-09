cmake_minimum_required(VERSION 3.15)
project(piano_companion_plugin)

if(MODC_BUILD_PLUGIN)
  add_subdirectory(../modcompose/juce juce)
  juce_add_plugin(PianoCompanion
    COMPANY_NAME "ModCompose"
    PLUGIN_NAME  "PianoCompanion"
    FORMATS      VST3 AU
    SOURCES
      PluginProcessor.cpp
      PluginProcessor.h
      PluginEditor.cpp
      PluginEditor.h
  )
  target_link_libraries(PianoCompanion PRIVATE pybind11::embed)
else()
  pybind11_add_module(piano_companion_plugin bindings/plugin.cpp)
  set_target_properties(piano_companion_plugin PROPERTIES PREFIX "" OUTPUT_NAME "piano_companion_plugin")
endif()
