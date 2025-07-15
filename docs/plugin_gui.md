# Plugin & GUI

This module provides a JUCE VST3 plugin and an Electron/React based GUI for controlling the generator.

## Build the Plugin

```bash
cd plugin
juce-cli build --target=VST3
make
```

Copy the resulting VST3 to your DAW's plugin directory.

## Build the GUI

```bash
cd gui
npm install
npm run build
```

Launch the packaged application to connect to the running plugin via WebSocket.
