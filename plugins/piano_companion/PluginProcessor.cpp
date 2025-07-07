#include "PluginProcessor.h"
#include "PluginEditor.h"

PianoCompanionAudioProcessor::PianoCompanionAudioProcessor()
    : state(*this, nullptr, juce::Identifier{"Params"},
             {
                 std::make_unique<juce::AudioParameterChoice>(
                     "TonePreset", "TonePreset",
                     juce::StringArray{"Default", "Bright", "Mellow"}, 0),
                 std::make_unique<juce::AudioParameterFloat>(
                     "Intensity", "Intensity", juce::NormalisableRange<float>(0.f, 2.f), 1.f),
                 std::make_unique<juce::AudioParameterFloat>(
                     "Temperature", "Temperature", juce::NormalisableRange<float>(0.f, 1.2f), 0.7f),
             }) {}

PianoCompanionAudioProcessor::~PianoCompanionAudioProcessor() {}

juce::AudioProcessorEditor* PianoCompanionAudioProcessor::createEditor() {
    return new PianoCompanionAudioProcessorEditor(*this);
}
