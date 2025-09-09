#include "PluginProcessor.h"
#include "PluginEditor.h"

VocalCompanionAudioProcessor::VocalCompanionAudioProcessor()
    : state(*this, nullptr, juce::Identifier{"Params"},
             {
                 std::make_unique<juce::AudioParameterChoice>(
                     "Backend", "Backend",
                     juce::StringArray{"synthv", "vocaloid", "onnx"}, 0),
                 std::make_unique<juce::AudioParameterString>(
                     "ModelPath", "ModelPath", ""),
                 std::make_unique<juce::AudioParameterBool>(
                     "EnableArticulation", "EnableArticulation", true),
                 std::make_unique<juce::AudioParameterFloat>(
                     "VibratoDepth", "VibratoDepth",
                     juce::NormalisableRange<float>(0.f, 1.f), 0.5f),
                 std::make_unique<juce::AudioParameterFloat>(
                     "VibratoRate", "VibratoRate",
                     juce::NormalisableRange<float>(1.f, 10.f), 5.f),
             }) {}

VocalCompanionAudioProcessor::~VocalCompanionAudioProcessor() {}

juce::AudioProcessorEditor* VocalCompanionAudioProcessor::createEditor() {
    return new VocalCompanionAudioProcessorEditor(*this);
}
