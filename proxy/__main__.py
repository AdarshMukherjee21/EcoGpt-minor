import logging

import uvicorn

from .app import create_app
from .config import LISTEN_HOST, LISTEN_PORT


def main() -> None:
    # Surface module INFO logs (e.g. proxy.chat_ui startup) — uvicorn.run()
    # configures only its own loggers, so without a root handler these
    # messages would be dropped silently.
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")
    # Production opts into BOTH sidecar managers (Headroom + the Open WebUI
    # chat UI). create_app defaults them to False so pytest and imports
    # never spawn subprocesses (see proxy/app.py).
    uvicorn.run(create_app(manage_sidecar=True, manage_chat_ui=True),
                host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")


if __name__ == "__main__":
    main()
