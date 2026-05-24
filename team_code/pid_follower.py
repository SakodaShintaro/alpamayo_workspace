import math

import carla
import numpy as np
from agents.navigation.controller import VehiclePIDController

from team_code.logger import get_logger

LOOKAHEAD_MIN_M = 4.0
LOOKAHEAD_MAX_M = 12.0
LOOKAHEAD_SPEED_GAIN = 0.4

TARGET_SPEED_MIN_KMH = 10.0
TARGET_SPEED_MAX_KMH = 35.0
TARGET_SPEED_EXTENT_GAIN = 1.0

ARGS_LAT = {"K_P": 1.1, "K_I": 0.02, "K_D": 0.15, "dt": 0.05}
ARGS_LON = {"K_P": 0.6, "K_I": 0.05, "K_D": 0.0, "dt": 0.05}

THROTTLE_MAX = 0.35
BRAKE_MAX = 1.0
SMOOTH_ALPHA = 0.25


def alpamayo_to_carla_local(wp_ego: np.ndarray) -> np.ndarray:
    out = np.asarray(wp_ego, dtype=np.float64).copy()
    out[:, 1] *= -1.0
    return out


def local_to_world(vehicle_tf: carla.Transform, wp_local: np.ndarray) -> np.ndarray:
    out = []
    for p in wp_local:
        loc = vehicle_tf.transform(carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        out.append([loc.x, loc.y, loc.z])
    return np.asarray(out, dtype=np.float64)


class _RawWaypoint:
    def __init__(self, transform: carla.Transform):
        self.transform = transform


class PIDTrajectoryFollower:
    def __init__(self, vehicle: carla.Vehicle):
        self.vehicle = vehicle
        self.pid = VehiclePIDController(
            vehicle,
            args_lateral=ARGS_LAT,
            args_longitudinal=ARGS_LON,
            max_throttle=THROTTLE_MAX,
            max_brake=BRAKE_MAX,
            max_steering=1.0,
        )
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0
        self._log = get_logger()

    def step(self, traj_ego_alpamayo: np.ndarray) -> carla.VehicleControl:
        traj_carla_local = alpamayo_to_carla_local(traj_ego_alpamayo)
        vehicle_tf = self.vehicle.get_transform()
        traj_world = local_to_world(vehicle_tf, traj_carla_local)

        velocity = self.vehicle.get_velocity()
        speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        lookahead = float(
            np.clip(
                LOOKAHEAD_MIN_M + speed_mps * LOOKAHEAD_SPEED_GAIN,
                LOOKAHEAD_MIN_M,
                LOOKAHEAD_MAX_M,
            )
        )

        ego_xy = np.array([vehicle_tf.location.x, vehicle_tf.location.y])
        dists = np.linalg.norm(traj_world[:, :2] - ego_xy[None, :], axis=1)
        idx = int(np.argmin(np.abs(dists - lookahead)))
        target_world = traj_world[idx]

        if idx + 1 < len(traj_world):
            dx = traj_world[idx + 1, 0] - traj_world[idx, 0]
            dy = traj_world[idx + 1, 1] - traj_world[idx, 1]
        else:
            dx = traj_world[idx, 0] - ego_xy[0]
            dy = traj_world[idx, 1] - ego_xy[1]
        target_yaw = math.degrees(math.atan2(dy, dx))

        target_tf = carla.Transform(
            carla.Location(
                x=float(target_world[0]), y=float(target_world[1]), z=float(target_world[2])
            ),
            carla.Rotation(yaw=target_yaw),
        )

        traj_extent = float(np.max(np.linalg.norm(traj_carla_local[:, :2], axis=1)))
        target_speed_kmh = float(
            np.clip(
                traj_extent * TARGET_SPEED_EXTENT_GAIN,
                TARGET_SPEED_MIN_KMH,
                TARGET_SPEED_MAX_KMH,
            )
        )

        raw = self.pid.run_step(target_speed_kmh, _RawWaypoint(target_tf))

        a = SMOOTH_ALPHA
        steer = a * raw.steer + (1.0 - a) * self._prev_steer
        throttle = a * raw.throttle + (1.0 - a) * self._prev_throttle
        brake = a * raw.brake + (1.0 - a) * self._prev_brake
        self._prev_steer, self._prev_throttle, self._prev_brake = steer, throttle, brake

        ctrl = carla.VehicleControl()
        ctrl.steer = float(steer)
        ctrl.throttle = float(throttle)
        ctrl.brake = float(brake)
        ctrl.hand_brake = False
        ctrl.manual_gear_shift = False
        self._log.info(
            f"PID speed={speed_mps:.2f}m/s lookahead={lookahead:.2f}m "
            f"target_speed={target_speed_kmh:.1f}km/h traj_extent={traj_extent:.2f}m "
            f"idx={idx} target_yaw={target_yaw:.1f} "
            f"raw(s={raw.steer:.3f},t={raw.throttle:.3f},b={raw.brake:.3f}) "
            f"out(s={ctrl.steer:.3f},t={ctrl.throttle:.3f},b={ctrl.brake:.3f})"
        )
        return ctrl
