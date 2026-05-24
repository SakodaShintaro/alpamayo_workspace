import math
from collections import deque

import numpy as np


class EgoHistoryBuffer:
    def __init__(self, capacity: int, stride: int):
        self.capacity = capacity
        self.stride = stride
        self._buf: deque[dict] = deque(maxlen=capacity * stride)

    def push(self, x: float, y: float, z: float, yaw_deg: float) -> None:
        self._buf.append({"x": x, "y": y, "z": z, "yaw": yaw_deg})

    def is_empty(self) -> bool:
        return len(self._buf) == 0

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._buf:
            raise RuntimeError("EgoHistoryBuffer is empty")

        samples = []
        for i in range(self.capacity):
            offset = -(1 + i * self.stride)
            if -offset > len(self._buf):
                samples.append(self._buf[0])
            else:
                samples.append(self._buf[offset])
        samples.reverse()

        current = samples[-1]
        current_pos = np.array([current["x"], current["y"], current["z"]], dtype=np.float64)
        current_yaw = math.radians(-current["yaw"])
        c, s = math.cos(current_yaw), math.sin(current_yaw)
        world_to_ego = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        xyz = np.zeros((self.capacity, 3), dtype=np.float32)
        rot = np.zeros((self.capacity, 3, 3), dtype=np.float32)
        for i, st in enumerate(samples):
            pos = np.array([st["x"], st["y"], st["z"]], dtype=np.float64)
            xyz[i] = world_to_ego @ (pos - current_pos)
            rel_yaw = math.radians(-st["yaw"]) - current_yaw
            cr, sr = math.cos(rel_yaw), math.sin(rel_yaw)
            rot[i] = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        return xyz, rot
