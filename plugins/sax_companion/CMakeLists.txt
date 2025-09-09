cmake_minimum_required(VERSION 3.15)
project(sax_companion_plugin)

if(MODC_BUILD_PLUGIN)
  add_subdirectory(juce)
  juce_add_plugin(SaxCompanion
    COMPANY_NAME "ModCompose"
    PLUGIN_NAME  "SaxCompanion"
    FORMATS      VST3 CLAP
    SOURCES
      PluginProcessor.cpp
      PluginProcessor.h
      PluginEditor.cpp
      PluginEditor.h
      plugin.cpp
      ringbuffer.h
  )
  target_link_libraries(SaxCompanion PRIVATE pybind11::embed)
else()
  # When MODC_BUILD_PLUGIN is OFF, build a lightweight Python module for tests.
  pybind11_add_module(sax_companion_plugin plugin.cpp)
  set_target_properties(sax_companion_plugin PROPERTIES PREFIX "" OUTPUT_NAME "sax_companion_plugin")
endif()
