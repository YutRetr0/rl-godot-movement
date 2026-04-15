import socket
import struct
from typing import Optional, Tuple

import numpy as np

MSG_RESET = 1
MSG_STEP = 2
MAX_OBS_DIM_SANITY_LIMIT = 4096


class RLGodotClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        if self.sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._ensure_connected()
        if seed is None:
            payload = struct.pack("<BB", MSG_RESET, 0)
        else:
            payload = struct.pack("<BBi", MSG_RESET, 1, int(seed))
        self._send(payload)
        obs, _, _ = self._recv_response()
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        self._ensure_connected()
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != 4:
            raise ValueError(f"Expected action shape (4,), got {action_arr.shape}")
        payload = struct.pack(
            "<B4f",
            MSG_STEP,
            float(action_arr[0]),
            float(action_arr[1]),
            float(action_arr[2]),
            float(action_arr[3]),
        )
        self._send(payload)
        return self._recv_response()

    def _ensure_connected(self) -> None:
        if self.sock is None:
            self.connect()

    def _send(self, payload: bytes) -> None:
        assert self.sock is not None
        self.sock.sendall(payload)

    def _recv_exact(self, nbytes: int) -> bytes:
        assert self.sock is not None
        chunks = []
        remaining = nbytes
        while remaining > 0:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Socket closed while receiving data")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _recv_response(self) -> Tuple[np.ndarray, float, bool]:
        header = self._recv_exact(4)
        (obs_dim,) = struct.unpack("<I", header)
        if obs_dim == 0 or obs_dim > MAX_OBS_DIM_SANITY_LIMIT:
            raise ValueError(f"Invalid observation dimension from server: {obs_dim}")

        obs_bytes = self._recv_exact(obs_dim * 4)
        obs = np.frombuffer(obs_bytes, dtype=np.float32).copy()

        reward_bytes = self._recv_exact(4)
        (reward,) = struct.unpack("<f", reward_bytes)

        done_bytes = self._recv_exact(1)
        done = bool(done_bytes[0])

        return obs, float(reward), done
