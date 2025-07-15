# Examples

## Quick start

1. Start the WebSocket bridge:
   ```bash
   python -m utilities.ws_bridge --config config/main_cfg.yml
   ```
2. In another terminal run the lofi client:
   ```bash
   python examples/lofi_ws_client.py
   ```
   It prints the list of note tokens received from the server.

