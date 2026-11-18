import logging

import uvicorn

from .app import create_app
from .config import LISTEN_HOST, LISTEN_PORT


def main() -> None:
    # Surface module INFO logs (e.g. proxy.chat_ui entry-URL harvest success /
    # failure) — uvicorn.run() configures only its own loggers, so without a
    # root handler these messages would be dropped silently.
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")
    # manage_chat_ui=True: production boot owns the Open WebUI sidecar
    # (chat UI served at /chat; see docs/chat-ui.md. Tests keep the
    # default False so no subprocess is ever spawned by pytest).
    uvicorn.run(create_app(manage_chat_ui=True), host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")


if __name__ == "__main__":
    main()
