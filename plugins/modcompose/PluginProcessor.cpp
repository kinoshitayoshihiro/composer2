#include "PluginProcessor.h"
#include "PluginEditor.h"

ModComposeAudioProcessor::ModComposeAudioProcessor() {}
ModComposeAudioProcessor::~ModComposeAudioProcessor() {}

juce::AudioProcessorEditor* ModComposeAudioProcessor::createEditor() {
    return new ModComposeAudioProcessorEditor(*this);
}

