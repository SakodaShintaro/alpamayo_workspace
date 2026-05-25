import json
import math
import os
import textwrap
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
from team_code.logger import configure_logger, get_logger
from team_code.pid_follower import PIDTrajectoryFollower, alpamayo_to_carla_local, local_to_world

MODEL_NAME = "nvidia/Alpamayo-1.5-10B"

IMG_WIDTH = 1280
IMG_HEIGHT = 720
NUM_FRAMES = 4
NUM_HISTORY = 16

HISTORY_STRIDE = 2

INFERENCE_INTERVAL_TICKS = 5

NAV_LOOKAHEAD_M = 50.0
ROAD_OPTION_TEXT = {
    RoadOption.LEFT: "Turn left",
    RoadOption.RIGHT: "Turn right",
    RoadOption.STRAIGHT: "Go straight at the intersection",
    RoadOption.CHANGELANELEFT: "Change lane to the left",
    RoadOption.CHANGELANERIGHT: "Change lane to the right",
}

CAMERAS = [
    {
        "id": "cam_front_left",
        "x": 1.0,
        "y": -0.5,
        "z": 2.4,
        "yaw": -60.0,
        "fov": 120,
        "alpamayo_idx": 0,
    },
    {
        "id": "cam_front_wide",
        "x": 1.5,
        "y": 0.0,
        "z": 2.4,
        "yaw": 0.0,
        "fov": 95,
        "alpamayo_idx": 1,
    },
    {
        "id": "cam_front_right",
        "x": 1.0,
        "y": 0.5,
        "z": 2.4,
        "yaw": 60.0,
        "fov": 120,
        "alpamayo_idx": 2,
    },
    {
        "id": "cam_front_tele",
        "x": 1.5,
        "y": 0.0,
        "z": 2.4,
        "yaw": 0.0,
        "fov": 30,
        "alpamayo_idx": 6,
    },
]

SPECTATOR_ID = "spectator"
SPECTATOR_WIDTH = 640
SPECTATOR_HEIGHT = 480
SPECTATOR_FOV = 90
SPECTATOR_INTERVAL_TICKS = 2

SPECTATOR_LOCAL_TF = carla.Transform(
    carla.Location(x=-8.0, y=0.0, z=5.0),
    carla.Rotation(roll=0.0, pitch=-20.0, yaw=0.0),
)


def get_entry_point() -> str:
    return "Alpamayo15Agent"


class Alpamayo15Agent(AutonomousAgent):
    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self._tick = 0
        self._ego_history = EgoHistoryBuffer(capacity=NUM_HISTORY, stride=HISTORY_STRIDE)
        self._frame_buffer: deque[dict] = deque(maxlen=NUM_FRAMES * HISTORY_STRIDE)
        self._cached_traj: np.ndarray | None = None
        self._cached_traj_world: np.ndarray | None = None
        self._cot_text: str = ""
        self._follower: PIDTrajectoryFollower | None = None
        self._dumped = False
        self._world_plan: list[tuple[carla.Transform, RoadOption]] = []
        self._log = get_logger()
        self._save_root = Path(os.environ.get("SAVE_PATH", "."))
        self._scenario_dir: Path | None = None
        self._spectator_dir: Path | None = None
        self._dump_dir: Path | None = None
        self._spectator_frame_idx = 0
        self._show_spectator = bool(os.environ.get("DISPLAY"))
        self._metric_info: dict[str, dict] = {}

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        save_name = path_to_conf_file.rsplit("+", 1)[-1] if "+" in path_to_conf_file else "default"
        self._scenario_dir = self._save_root / save_name
        self._spectator_dir = self._scenario_dir / "spectator"
        self._spectator_dir.mkdir(parents=True, exist_ok=True)
        self._dump_dir = self._scenario_dir / "input_images"
        configure_logger(self._scenario_dir)
        self._log.info(f"scenario output dir: {self._scenario_dir}")
        self._log.info(f"global plan: {len(self._world_plan)} waypoints")
        if not self._show_spectator:
            self._log.info("DISPLAY not set, spectator window disabled")
        self._log.info(f"loading {MODEL_NAME} ...")
        self.model = Alpamayo1_5.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
        self.processor = helper.get_processor(self.model.tokenizer)
        self._log.info("model loaded")

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._world_plan = list(global_plan_world_coord)

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
                "x": c["x"],
                "y": c["y"],
                "z": c["z"],
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": c["yaw"],
                "width": IMG_WIDTH,
                "height": IMG_HEIGHT,
                "fov": c["fov"],
            }
            for c in CAMERAS
        ]
        spectator = {
            "type": "sensor.camera.rgb",
            "id": SPECTATOR_ID,
            "x": SPECTATOR_LOCAL_TF.location.x,
            "y": SPECTATOR_LOCAL_TF.location.y,
            "z": SPECTATOR_LOCAL_TF.location.z,
            "roll": SPECTATOR_LOCAL_TF.rotation.roll,
            "pitch": SPECTATOR_LOCAL_TF.rotation.pitch,
            "yaw": SPECTATOR_LOCAL_TF.rotation.yaw,
            "width": SPECTATOR_WIDTH,
            "height": SPECTATOR_HEIGHT,
            "fov": SPECTATOR_FOV,
        }
        return policy_cams + [spectator]

    def run_step(self, input_data, timestamp):
        if self._follower is None:
            self._follower = PIDTrajectoryFollower(self.hero_actor)

        self._metric_info[f"{self._tick:08d}"] = self.get_metric_info()

        tf = self.hero_actor.get_transform()
        self._ego_history.push(tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw)

        frame = {}
        for c in CAMERAS:
            bgra = input_data[c["id"]][1]
            rgb = bgra[:, :, [2, 1, 0]]
            frame[c["id"]] = rgb
        self._frame_buffer.append(frame)

        if self._tick % SPECTATOR_INTERVAL_TICKS == 0:
            bgr = input_data[SPECTATOR_ID][1][:, :, :3].copy()
            self._draw_overlay(bgr)
            cv2.imwrite(
                str(self._spectator_dir / f"frame_{self._spectator_frame_idx:08d}.png"), bgr
            )
            self._spectator_frame_idx += 1
            if self._show_spectator:
                cv2.imshow(SPECTATOR_ID, bgr)
                cv2.waitKey(1)

        ready = len(self._frame_buffer) == NUM_FRAMES * HISTORY_STRIDE
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
        sampled_frames = list(self._frame_buffer)[HISTORY_STRIDE - 1 :: HISTORY_STRIDE]
        images = np.stack(
            [np.stack([frame[c["id"]] for c in CAMERAS], axis=0) for frame in sampled_frames],
            axis=1,
        )
        if not self._dumped:
            self._dumped = True
            dump_dir = self._dump_dir
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
        messages = helper.create_message(
            image_tensor.flatten(0, 1), camera_indices=camera_indices, nav_text=nav_text
        )
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
            pred_xyz, _pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                diffusion_kwargs={"inference_step": 10},
                max_generation_length=256,
                return_extra=True,
            )

        traj = pred_xyz[0, 0, 0].float().cpu().numpy()
        self._log.info(
            f"traj ego_frame: wp0={traj[0]} wp10={traj[10]} wp30={traj[30]} wp63={traj[63]}"
        )

        ego_tf_at_inference = self.hero_actor.get_transform()
        self._cached_traj_world = local_to_world(ego_tf_at_inference, alpamayo_to_carla_local(traj))

        cot_arr = extra.get("cot") if isinstance(extra, dict) else None
        if cot_arr is not None:
            self._cot_text = str(np.asarray(cot_arr).reshape(-1)[0])
            self._log.info(f"cot: {self._cot_text}")
        return traj

    def _draw_overlay(self, bgr: np.ndarray) -> None:
        if self._cached_traj_world is not None:
            ego_mat = np.array(self.hero_actor.get_transform().get_matrix())
            spec_local_mat = np.array(SPECTATOR_LOCAL_TF.get_matrix())
            spec_world_mat = ego_mat @ spec_local_mat
            spec_inv_mat = np.linalg.inv(spec_world_mat)
            fx = fy = SPECTATOR_WIDTH / (2 * math.tan(math.radians(SPECTATOR_FOV) / 2))
            cx, cy = SPECTATOR_WIDTH / 2, SPECTATOR_HEIGHT / 2
            for wp in self._cached_traj_world:
                p_h = np.array([wp[0], wp[1], wp[2], 1.0])
                p_cam = spec_inv_mat @ p_h
                if p_cam[0] <= 0.1:
                    continue
                u = int(fx * p_cam[1] / p_cam[0] + cx)
                v = int(fy * (-p_cam[2]) / p_cam[0] + cy)
                if 0 <= u < SPECTATOR_WIDTH and 0 <= v < SPECTATOR_HEIGHT:
                    cv2.circle(bgr, (u, v), 3, (0, 255, 0), -1)

        if self._cot_text:
            lines = textwrap.wrap(self._cot_text, width=80)[-3:]
            line_h = 18
            y0 = SPECTATOR_HEIGHT - line_h * len(lines) - 6
            cv2.rectangle(
                bgr,
                (0, y0 - 4),
                (SPECTATOR_WIDTH, SPECTATOR_HEIGHT),
                (0, 0, 0),
                -1,
            )
            for i, line in enumerate(lines):
                cv2.putText(
                    bgr,
                    line,
                    (8, y0 + line_h * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    def destroy(self):
        if self._scenario_dir is not None and self._metric_info:
            with open(self._scenario_dir / "metric_info.json", "w") as f:
                json.dump(self._metric_info, f)
        if self._show_spectator:
            cv2.destroyAllWindows()
        self.model = None
        torch.cuda.empty_cache()
