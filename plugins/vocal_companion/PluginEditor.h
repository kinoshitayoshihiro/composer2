#pragma once
#include <juce_gui_extra/juce_gui_extra.h>

class VocalCompanionAudioProcessor;

class VocalCompanionAudioProcessorEditor : public juce::AudioProcessorEditor {
public:
    explicit VocalCompanionAudioProcessorEditor(VocalCompanionAudioProcessor&);
    ~VocalCompanionAudioProcessorEditor() override;

    void paint(juce::Graphics&) override {}
    void resized() override {}

private:
    VocalCompanionAudioProcessor& processor;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VocalCompanionAudioProcessorEditor)
};
