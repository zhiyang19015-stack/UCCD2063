"""Gym-like wrapper around the TCP connection to Godot's rl_environment.gd.

Speaks the cmd-based protocol:
  {"cmd": "ping"}              -> {"cmd": "pong"}
  {"cmd": "reset"}             -> {"cmd": "reset", "state": [...], "info": {...}}
  {"cmd": "step", "action": n} -> {"cmd": "step", "state": [...], "reward": f, "done": bool, "info": {...}}
  {"cmd": "state"}             -> {"cmd": "state", "state": [...], "info": {...}}
  {"cmd": "close"}             -> {"cmd": "close", "status": "ok"}
"""

from tcp_server import GodotConnection


class GodotEnv:
    def __init__(self, conn: GodotConnection):
        self._conn = conn

    def ping(self) -> bool:
        self._conn.send({"cmd": "ping"})
        reply = self._conn.recv_message()
        return reply.get("cmd") == "pong"

    def reset(self) -> tuple:
        self._conn.send({"cmd": "reset"})
        reply = self._conn.recv_message()
        return reply["state"], reply.get("info", {})

    def step(self, action: int) -> tuple:
        self._conn.send({"cmd": "step", "action": action})
        reply = self._conn.recv_message()
        return reply["state"], reply["reward"], reply["done"], reply.get("info", {})

    def peek_state(self) -> tuple:
        self._conn.send({"cmd": "state"})
        reply = self._conn.recv_message()
        return reply["state"], reply.get("info", {})

    def close(self) -> None:
        try:
            self._conn.send({"cmd": "close"})
            self._conn.recv_message()
        except (ConnectionError, OSError):
            pass
