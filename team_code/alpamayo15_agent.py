from collections import deque

import carla
import numpy as np
import torch
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from team_code.ego_history import EgoHistoryBuffer
from team_code.pid_follower import PIDTrajectoryFollower

MODEL_NAME = "nvidia/Alpamayo-1.5-10B"

IMG_WIDTH = 1280
IMG_HEIGHT = 720
NUM_FRAMES = 4
NUM_HISTORY = 16

INFERENCE_INTERVAL_TICKS = 10

CAMERAS = [
    {"id": "cam_front_left",  "x": 1.0, "y": -0.5, "z": 2.4, "yaw": -60.0, "fov": 120},
    {"id": "cam_front_wide",  "x": 1.5, "y":  0.0, "z": 2.4, "yaw":   0.0, "fov":  95},
    {"id": "cam_front_right", "x": 1.0, "y":  0.5, "z": 2.4, "yaw":  60.0, "fov": 120},
    {"id": "cam_front_tele",  "x": 1.5, "y":  0.0, "z": 2.4, "yaw":   0.0, "fov":  30},
]


def get_entry_point() -> str:
    return "Alpamayo15Agent"


class Alpamayo15Agent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self._tick = 0
        self._ego_history = EgoHistoryBuffer(capacity=NUM_HISTORY)
        self._frame_buffer: deque[dict] = deque(maxlen=NUM_FRAMES)
        self._cached_traj: np.ndarray | None = None
        self._follower: PIDTrajectoryFollower | None = None

        print(f"[Alpamayo15Agent] loading {MODEL_NAME} ...", flush=True)
        self.model = Alpamayo1_5.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
        self.processor = helper.get_processor(self.model.tokenizer)
        print("[Alpamayo15Agent] model loaded", flush=True)

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "id": c["id"],
                "x": c["x"], "y": c["y"], "z": c["z"],
                "roll": 0.0, "pitch": 0.0, "yaw": c["yaw"],
                "width": IMG_WIDTH, "height": IMG_HEIGHT, "fov": c["fov"],
            }
            for c in CAMERAS
        ]

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
        image_tensor = torch.from_numpy(images).permute(0, 1, 4, 2, 3).contiguous()

        xyz, rot = self._ego_history.snapshot()
        hist_xyz = torch.from_numpy(xyz).float().unsqueeze(0).unsqueeze(0)
        hist_rot = torch.from_numpy(rot).float().unsqueeze(0).unsqueeze(0)

        messages = helper.create_message(image_tensor.flatten(0, 1), camera_indices=None, nav_text=None)
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
        return traj

    def destroy(self):
        self.model = None
        torch.cuda.empty_cache()
