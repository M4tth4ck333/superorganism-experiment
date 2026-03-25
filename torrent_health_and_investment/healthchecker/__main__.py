import argparse
import asyncio
import threading
import time

from healthchecker.gui import run_gui
from healthchecker.liberation_service import LiberationService
from healthchecker.db import init_db


def main():
    parser = argparse.ArgumentParser(description="SwarmHealth - Creative Commons Torrent Health Checker")
    parser.add_argument(
        "--key-file",
        default=None,
        help="Path to IPv8 key file (default: liberation_key.pem)"
    )
    args = parser.parse_args()

    init_db()
    _, liberation_service = run_liberation_in_thread(args.key_file)
    print("Waiting for IPv8 network to initialize...")
    time.sleep(10)
    run_gui(seedbox_fleet=liberation_service.seedbox_fleet)


def run_liberation_in_thread(key_file: str = None):
    """Run the liberation service in a background thread with its own event loop.
    Returns (thread, service) — service.seedbox_fleet is readable from any thread.
    """
    service = LiberationService(key_file=key_file or "liberation_key.pem")

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(service.start())
            loop.run_forever()
        except Exception as e:
            print(f"Liberation service error: {e}")
        finally:
            loop.run_until_complete(service.stop())
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, service


if __name__ == "__main__":
    main()
