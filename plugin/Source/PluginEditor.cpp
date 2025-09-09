#include "PluginEditor.h"
#include "PluginProcessor.h"

ModComposerAudioProcessorEditor::ModComposerAudioProcessorEditor(ModComposerAudioProcessor& p)
    : AudioProcessorEditor(&p), processor(p)
{
    setSize(400, 300);
}

ModComposerAudioProcessorEditor::~ModComposerAudioProcessorEditor() = default;

void ModComposerAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
}

void ModComposerAudioProcessorEditor::resized() {}
