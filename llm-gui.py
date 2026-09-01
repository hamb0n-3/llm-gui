#!/usr/bin/env python3
from __future__ import annotations

from config import SERVER_HOST, SERVER_PORT
from ui import build_ui


def main() -> None:
    demo = build_ui()
    demo.queue()  # needed for streaming cancel
    try:
        demo.launch(server_name=SERVER_HOST, server_port=SERVER_PORT)
    except OSError:  # configured port busy — let Gradio pick a free one
        demo.launch(server_name=SERVER_HOST, server_port=None)


if __name__ == "__main__":
    main()
