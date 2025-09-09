#include "PluginEditor.h"
#include "PluginProcessor.h"

SaxCompanionAudioProcessorEditor::SaxCompanionAudioProcessorEditor(SaxCompanionAudioProcessor& p)
    : juce::AudioProcessorEditor(p), processor(p) {
    addAndMakeVisible(growlButton);
    addAndMakeVisible(altissimoButton);
    setSize(400, 100);
}

SaxCompanionAudioProcessorEditor::~SaxCompanionAudioProcessorEditor() {}

void SaxCompanionAudioProcessorEditor::resized() {
    auto area = getLocalBounds();
    growlButton.setBounds(area.removeFromLeft(190).reduced(5));
    altissimoButton.setBounds(area.reduced(5));
}

