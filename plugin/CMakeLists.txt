cmake_minimum_required(VERSION 3.15)
project(ModComposerPlugin VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

find_package(JUCE CONFIG REQUIRED)

juce_add_plugin(ModComposerPlugin
    COMPANY_NAME "Composer2"
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    PLUGIN_MANUFACTURER_CODE MCpl
    PLUGIN_CODE MCpl
    FORMATS VST3
    PRODUCT_NAME "ModComposerPlugin")

juce_generate_juce_header(ModComposerPlugin)

target_sources(ModComposerPlugin PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp)

target_link_libraries(ModComposerPlugin PRIVATE
    juce::juce_audio_utils)
