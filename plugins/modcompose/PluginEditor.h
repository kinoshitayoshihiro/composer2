#pragma once
#include <juce_gui_extra/juce_gui_extra.h>

class ModComposeAudioProcessor;

class ModComposeAudioProcessorEditor : public juce::AudioProcessorEditor {
public:
    explicit ModComposeAudioProcessorEditor(ModComposeAudioProcessor&);
    ~ModComposeAudioProcessorEditor() override;

    void paint(juce::Graphics&) override {}
    void resized() override {}

private:
    ModComposeAudioProcessor& processor;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModComposeAudioProcessorEditor)
};
