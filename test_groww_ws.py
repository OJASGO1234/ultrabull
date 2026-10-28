
import asyncio
import websockets

async def main():
    url = "ws://127.0.0.1:8000/api/live/ws?symbols=RELIANCE.NS"

    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            print("Connected to UltraBull WebSocket!")

            try:
                message = await asyncio.wait_for(ws.recv(), timeout=20)
                print("Received message:")
                print(message)
            except asyncio.TimeoutError:
                print("Connected, but no message arrived within 20 seconds.")
                print("This can happen when the market is closed.")

    except Exception as exc:
        print(f"WebSocket test failed: {type(exc).__name__}: {exc}")

asyncio.run(main())
