#pragma once
#include <juce_gui_extra/juce_gui_extra.h>

class SaxCompanionAudioProcessor;

class SaxCompanionAudioProcessorEditor : public juce::AudioProcessorEditor {
public:
    explicit SaxCompanionAudioProcessorEditor(SaxCompanionAudioProcessor&);
    ~SaxCompanionAudioProcessorEditor() override;

    void paint(juce::Graphics&) override {}
    void resized() override {}

private:
    juce::TextButton growlButton{"Growl"};
    juce::TextButton altissimoButton{"Altissimo"};
    SaxCompanionAudioProcessor& processor;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SaxCompanionAudioProcessorEditor)
};
