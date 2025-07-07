#include "PluginEditor.h"
#include "PluginProcessor.h"

PianoCompanionAudioProcessorEditor::PianoCompanionAudioProcessorEditor(PianoCompanionAudioProcessor& p)
    : juce::AudioProcessorEditor(p), processor(p) {
    setSize(400, 300);
}

PianoCompanionAudioProcessorEditor::~PianoCompanionAudioProcessorEditor() {}
