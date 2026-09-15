"""Private TCP-to-Unix-socket relay used by rootless Podman sandboxes."""

from __future__ import annotations

import os
import socket
import socketserver
import sys
import threading
from contextlib import suppress
from pathlib import Path


class _RelayHandler(socketserver.BaseRequestHandler):
    server: _RelayServer

    def handle(self) -> None:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect(os.fspath(self.server.unix_socket))
            upload = threading.Thread(target=self._copy, args=(self.request, upstream))
            upload.start()
            self._copy(upstream, self.request)
            upload.join()
        finally:
            upstream.close()

    @staticmethod
    def _copy(source: socket.socket, target: socket.socket) -> None:
        try:
            while chunk := source.recv(64 * 1024):
                target.sendall(chunk)
        except OSError:
            pass
        with suppress(OSError):
            target.shutdown(socket.SHUT_WR)


class _RelayServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False

    unix_socket: Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: _tcp_unix_relay.py SOCKET")
    with _RelayServer(("0.0.0.0", 0), _RelayHandler) as server:
        server.unix_socket = Path(sys.argv[1])
        print(server.server_address[1], flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
