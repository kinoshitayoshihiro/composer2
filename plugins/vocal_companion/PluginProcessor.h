#pragma once
#include <juce_audio_processors/juce_audio_processors.h>

class VocalCompanionAudioProcessor : public juce::AudioProcessor {
public:
    VocalCompanionAudioProcessor();
    ~VocalCompanionAudioProcessor() override;

    void prepareToPlay(double, int) override {}
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout&) const override { return true; }
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override {}

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "VocalCompanion"; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    juce::AudioProcessorValueTreeState state;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VocalCompanionAudioProcessor)
};
