import os
from collections import deque
from pathlib import Path

import carla
import cv2
import numpy as np
import torch
from agents.navigation.local_planner import RoadOption
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from team_code.ego_history import EgoHistoryBuffer
from team_code.logger import get_logger
from team_code.pid_follower import PIDTrajectoryFollower

MODEL_NAME = "nvidia/Alpamayo-1.5-10B"

IMG_WIDTH = 1280
IMG_HEIGHT = 720
NUM_FRAMES = 4
NUM_HISTORY = 16

INFERENCE_INTERVAL_TICKS = 10

NAV_LOOKAHEAD_M = 50.0
ROAD_OPTION_TEXT = {
    RoadOption.LEFT: "Turn left",
    RoadOption.RIGHT: "Turn right",
    RoadOption.STRAIGHT: "Go straight at the intersection",
    RoadOption.CHANGELANELEFT: "Change lane to the left",
    RoadOption.CHANGELANERIGHT: "Change lane to the right",
}

CAMERAS = [
    {"id": "cam_front_left",  "x": 1.0, "y": -0.5, "z": 2.4, "yaw": -60.0, "fov": 120, "alpamayo_idx": 0},
    {"id": "cam_front_wide",  "x": 1.5, "y":  0.0, "z": 2.4, "yaw":   0.0, "fov":  95, "alpamayo_idx": 1},
    {"id": "cam_front_right", "x": 1.0, "y":  0.5, "z": 2.4, "yaw":  60.0, "fov": 120, "alpamayo_idx": 2},
    {"id": "cam_front_tele",  "x": 1.5, "y":  0.0, "z": 2.4, "yaw":   0.0, "fov":  30, "alpamayo_idx": 6},
]

SPECTATOR_ID = "spectator"
SPECTATOR_WIDTH = 640
SPECTATOR_HEIGHT = 480
SPECTATOR_INTERVAL_TICKS = 2


def get_entry_point() -> str:
    return "Alpamayo15Agent"


class Alpamayo15Agent(AutonomousAgent):
    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self._tick = 0
        self._ego_history = EgoHistoryBuffer(capacity=NUM_HISTORY)
        self._frame_buffer: deque[dict] = deque(maxlen=NUM_FRAMES)
        self._cached_traj: np.ndarray | None = None
        self._follower: PIDTrajectoryFollower | None = None
        self._dumped = False
        self._world_plan: list[tuple[carla.Transform, RoadOption]] = []
        self._log = get_logger()
        self._spectator_dir = Path(os.environ.get("SAVE_PATH", ".")) / "spectator"
        self._spectator_dir.mkdir(parents=True, exist_ok=True)
        self._spectator_frame_idx = 0

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self._log.info(f"loading {MODEL_NAME} ...")
        self.model = Alpamayo1_5.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
        self.processor = helper.get_processor(self.model.tokenizer)
        self._log.info("model loaded")

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._world_plan = list(global_plan_world_coord)
        self._log.info(f"global plan received: {len(self._world_plan)} waypoints")

    def _nav_text(self, ego_loc: carla.Location) -> str:
        if not self._world_plan:
            return "Continue straight"
        nearest_idx = min(
            range(len(self._world_plan)),
            key=lambda i: self._world_plan[i][0].location.distance(ego_loc),
        )
        accumulated = 0.0
        prev_loc = ego_loc
        for i in range(nearest_idx, len(self._world_plan)):
            tf, opt = self._world_plan[i]
            accumulated += prev_loc.distance(tf.location)
            prev_loc = tf.location
            if opt in ROAD_OPTION_TEXT:
                return f"{ROAD_OPTION_TEXT[opt]} in {accumulated:.0f}m"
            if accumulated > NAV_LOOKAHEAD_M:
                break
        return "Continue straight"

    def sensors(self):
        policy_cams = [
            {
                "type": "sensor.camera.rgb",
                "id": c["id"],
                "x": c["x"], "y": c["y"], "z": c["z"],
                "roll": 0.0, "pitch": 0.0, "yaw": c["yaw"],
                "width": IMG_WIDTH, "height": IMG_HEIGHT, "fov": c["fov"],
            }
            for c in CAMERAS
        ]
        spectator = {
            "type": "sensor.camera.rgb",
            "id": SPECTATOR_ID,
            "x": -8.0, "y": 0.0, "z": 5.0,
            "roll": 0.0, "pitch": -20.0, "yaw": 0.0,
            "width": SPECTATOR_WIDTH, "height": SPECTATOR_HEIGHT, "fov": 90,
        }
        return policy_cams + [spectator]

    def run_step(self, input_data, timestamp):
        if self._follower is None:
            self._follower = PIDTrajectoryFollower(self.hero_actor)

        tf = self.hero_actor.get_transform()
        self._ego_history.push(tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw)

        frame = {}
        for c in CAMERAS:
            bgra = input_data[c["id"]][1]
            rgb = bgra[:, :, [2, 1, 0]]
            frame[c["id"]] = rgb
        self._frame_buffer.append(frame)

        if self._tick % SPECTATOR_INTERVAL_TICKS == 0:
            bgr = input_data[SPECTATOR_ID][1][:, :, :3]
            cv2.imwrite(str(self._spectator_dir / f"frame_{self._spectator_frame_idx:08d}.png"), bgr)
            self._spectator_frame_idx += 1

        ready = len(self._frame_buffer) == NUM_FRAMES
        if ready and (self._cached_traj is None or self._tick % INFERENCE_INTERVAL_TICKS == 0):
            self._cached_traj = self._run_inference()

        self._tick += 1

        if self._cached_traj is None:
            ctrl = carla.VehicleControl()
            ctrl.throttle = 0.0
            ctrl.brake = 0.0
            ctrl.steer = 0.0
            return ctrl

        return self._follower.step(self._cached_traj)

    def _run_inference(self) -> np.ndarray:
        images = np.stack(
            [np.stack([frame[c["id"]] for c in CAMERAS], axis=0) for frame in self._frame_buffer],
            axis=1,
        )
        if not self._dumped:
            self._dumped = True
            dump_dir = Path(os.environ.get("SAVE_PATH", ".")) / "input_images"
            dump_dir.mkdir(parents=True, exist_ok=True)
            for ci, c in enumerate(CAMERAS):
                cam_dir = dump_dir / c["id"]
                cam_dir.mkdir(parents=True, exist_ok=True)
                for fi in range(images.shape[1]):
                    cv2.imwrite(str(cam_dir / f"frame{fi:08d}.png"), images[ci, fi, :, :, ::-1])
            self._log.info(f"dumped first-inference camera frames to {dump_dir}")
        image_tensor = torch.from_numpy(images).permute(0, 1, 4, 2, 3).contiguous()

        xyz, rot = self._ego_history.snapshot()
        hist_xyz = torch.from_numpy(xyz).float().unsqueeze(0).unsqueeze(0)
        hist_rot = torch.from_numpy(rot).float().unsqueeze(0).unsqueeze(0)

        nav_text = self._nav_text(self.hero_actor.get_location())
        self._log.info(f"nav_text: {nav_text}")
        camera_indices = torch.tensor([c["alpamayo_idx"] for c in CAMERAS], dtype=torch.long)
        messages = helper.create_message(image_tensor.flatten(0, 1), camera_indices=camera_indices, nav_text=nav_text)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = helper.to_device(
            {"tokenized_data": inputs, "ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot},
            "cuda",
        )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _pred_rot = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                diffusion_kwargs={"inference_step": 10},
                max_generation_length=256,
            )

        traj = pred_xyz[0, 0, 0].float().cpu().numpy()
        self._log.info(
            f"traj ego_frame: wp0={traj[0]} wp10={traj[10]} wp30={traj[30]} wp63={traj[63]}"
        )
        return traj

    def destroy(self):
        self.model = None
        torch.cuda.empty_cache()
