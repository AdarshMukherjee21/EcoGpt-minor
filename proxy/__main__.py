import uvicorn

from .app import create_app
from .config import LISTEN_HOST, LISTEN_PORT


def main() -> None:
    # manage_chat_ui=True: production boot owns the dsh-harness sidecar
    # (vendored web UI served at /chat; see docs/chat-ui.md. Tests keep the
    # default False so no subprocess is ever spawned by pytest).
    uvicorn.run(create_app(manage_chat_ui=True), host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")


if __name__ == "__main__":
    main()
