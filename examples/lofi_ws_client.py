import asyncio
import json

import websockets


async def main() -> None:
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(json.dumps({"chord": [60, 64, 67], "bars_context": 2}))
        async for message in ws:
            notes = json.loads(message)
            print("received", notes)
            break


if __name__ == "__main__":
    asyncio.run(main())
