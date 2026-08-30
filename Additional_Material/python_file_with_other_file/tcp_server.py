"""Minimal TCP transport for talking to Godot's tcp_bridge.gd.

Godot connects to us as a TCP client; we're the server. Messages are
newline-delimited JSON in both directions -- no external dependencies.
"""

import json
import socket


class GodotConnection:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buffer = b""

    def send(self, message: dict) -> None:
        data = (json.dumps(message) + "\n").encode("utf-8")
        self.sock.sendall(data)

    def recv_message(self) -> dict:
        while b"\n" not in self._buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Godot closed the TCP connection")
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return json.loads(line.decode("utf-8"))

    def close(self) -> None:
        self.sock.close()


def wait_for_godot(host: str = "127.0.0.1", port: int = 5555) -> GodotConnection:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[tcp_server] Waiting for Godot to connect on {host}:{port} ... (run the Godot project now)")
    conn, addr = server.accept()
    print(f"[tcp_server] Godot connected from {addr}")
    server.close()
    return GodotConnection(conn)
