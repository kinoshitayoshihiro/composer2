#include "PluginProcessor.h"
#include "PluginEditor.h"

SaxCompanionAudioProcessor::SaxCompanionAudioProcessor() {}
SaxCompanionAudioProcessor::~SaxCompanionAudioProcessor() {}

juce::AudioProcessorEditor* SaxCompanionAudioProcessor::createEditor() {
    return new SaxCompanionAudioProcessorEditor(*this);
}

