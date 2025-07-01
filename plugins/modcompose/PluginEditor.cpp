#include "PluginEditor.h"
#include "PluginProcessor.h"

ModComposeAudioProcessorEditor::ModComposeAudioProcessorEditor(ModComposeAudioProcessor& p)
    : juce::AudioProcessorEditor(p), processor(p) {
    setSize(400, 300);
}

ModComposeAudioProcessorEditor::~ModComposeAudioProcessorEditor() {}

