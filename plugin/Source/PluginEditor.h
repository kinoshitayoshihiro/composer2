#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class ModComposerAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit ModComposerAudioProcessorEditor(ModComposerAudioProcessor&);
    ~ModComposerAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    ModComposerAudioProcessor& processor;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModComposerAudioProcessorEditor)
};
