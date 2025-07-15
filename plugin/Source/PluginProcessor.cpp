#include "PluginProcessor.h"
#include "PluginEditor.h"

ModComposerAudioProcessor::ModComposerAudioProcessor()
    : AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, "PARAMETERS", createParameters())
{
}

ModComposerAudioProcessor::~ModComposerAudioProcessor() = default;

void ModComposerAudioProcessor::prepareToPlay(double, int) {}
void ModComposerAudioProcessor::releaseResources() {}

void ModComposerAudioProcessor::processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) {}

juce::AudioProcessorEditor* ModComposerAudioProcessor::createEditor() {
    return new ModComposerAudioProcessorEditor(*this);
}

bool ModComposerAudioProcessor::hasEditor() const { return true; }

const juce::String ModComposerAudioProcessor::getName() const { return JucePlugin_Name; }

void ModComposerAudioProcessor::getStateInformation(juce::MemoryBlock&) {}
void ModComposerAudioProcessor::setStateInformation(const void*, int) {}

juce::AudioProcessorValueTreeState::ParameterLayout ModComposerAudioProcessor::createParameters() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    params.push_back(std::make_unique<juce::AudioParameterInt>("tempo", "Tempo", 40, 240, 120));
    juce::StringArray styles { "Classical", "Jazz", "Rock" };
    params.push_back(std::make_unique<juce::AudioParameterChoice>("style", "Style", styles, 0));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("density", "Density", juce::NormalisableRange<float>(0.0f, 1.0f), 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterString>("section", "Section", "Verse"));
    return { params.begin(), params.end() };
}
