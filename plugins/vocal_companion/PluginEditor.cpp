#include "PluginEditor.h"
#include "PluginProcessor.h"

VocalCompanionAudioProcessorEditor::VocalCompanionAudioProcessorEditor(VocalCompanionAudioProcessor& p)
    : juce::AudioProcessorEditor(p), processor(p) {
    setSize(400, 300);
}

VocalCompanionAudioProcessorEditor::~VocalCompanionAudioProcessorEditor() {}
