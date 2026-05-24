import math
from collections import deque

import numpy as np


class EgoHistoryBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque[dict] = deque(maxlen=capacity)

    def push(self, x: float, y: float, z: float, yaw_deg: float) -> None:
        self._buf.append({"x": x, "y": y, "z": z, "yaw": yaw_deg})

    def is_empty(self) -> bool:
        return len(self._buf) == 0

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._buf:
            raise RuntimeError("EgoHistoryBuffer is empty")

        while len(self._buf) < self.capacity:
            self._buf.appendleft(self._buf[0])

        current = self._buf[-1]
        current_pos = np.array([current["x"], current["y"], current["z"]], dtype=np.float64)
        current_yaw = math.radians(-current["yaw"])
        c, s = math.cos(current_yaw), math.sin(current_yaw)
        world_to_ego = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        xyz = np.zeros((self.capacity, 3), dtype=np.float32)
        rot = np.zeros((self.capacity, 3, 3), dtype=np.float32)
        for i, st in enumerate(self._buf):
            pos = np.array([st["x"], st["y"], st["z"]], dtype=np.float64)
            xyz[i] = world_to_ego @ (pos - current_pos)
            rel_yaw = math.radians(-st["yaw"]) - current_yaw
            cr, sr = math.cos(rel_yaw), math.sin(rel_yaw)
            rot[i] = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        print(f"[EgoHistoryBuffer] snapshot: xyz.shape={xyz.shape}, rot.shape={rot.shape}")
        for i in range(self.capacity):
            print(f"  [{i}] xyz={xyz[i]}, rot=\n{rot[i]}")
        return xyz, rot
