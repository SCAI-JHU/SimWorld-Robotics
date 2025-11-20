"""
HumanInterface: simple human-role controller built on top of UnrealCV utilities.

Features
- Initialize with an UnrealCV instance and an actor name representing the human in the simulator
- Provide simple, user-friendly methods to move (forward/back/left/right) and turn (left/right)
- Use defaults from config/action_config.json
- Apply a cooldown window of duration*2 after each command to avoid conflicts

Example
    from simworld_gym.utils.unrealcv_basic import UnrealCV
    from simworld_gym.utils.human_interface import HumanInterface

    ucv = UnrealCV(port=9900, ip="127.0.0.1", resolution=(320, 240))
    human = HumanInterface(ucv, actor_name="Human01")

    human.move_forward()    # move using defaults
    human.turn_left()       # rotate using defaults

"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .unrealcv_basic import UnrealCV


@dataclass(frozen=True)
class MovementDefaults:
    speed: float
    duration: float
    dir_forward: int
    dir_backward: int
    dir_left: int
    dir_right: int


@dataclass(frozen=True)
class RotationDefaults:
    duration: float
    angle_left: float
    angle_right: float
    dir_left: int
    dir_right: int


class HumanInterface:
    """Convenient wrapper to control a human-role actor in the simulator.

    Methods are intentionally small and explicit to make higher-level scripting easy.
    A cooldown prevents overlapping commands (busy for duration*2 after each call).
    """

    def __init__(self, unrealcv: UnrealCV, actor_name: str, action_config_path: Optional[str] = None):
        self._ucv = unrealcv
        self.actor_name = actor_name

        if action_config_path is None:
            # Default to utils/../config/action_config.json
            action_config_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "config", "action_config.json")
            )

        self._movement, self._rotation = self._load_defaults(action_config_path)

        self._busy_until = 0.0
        self._lock = threading.Lock()

    # ------------- Public API -------------
    def move_forward(self, speed: Optional[float] = None, duration: Optional[float] = None) -> bool:
        return self._move(self._movement.dir_forward, speed, duration)

    def move_backward(self, speed: Optional[float] = None, duration: Optional[float] = None) -> bool:
        return self._move(self._movement.dir_backward, speed, duration)

    def move_left(self, speed: Optional[float] = None, duration: Optional[float] = None) -> bool:
        return self._move(self._movement.dir_left, speed, duration)

    def move_right(self, speed: Optional[float] = None, duration: Optional[float] = None) -> bool:
        return self._move(self._movement.dir_right, speed, duration)

    def turn_left(self, angle: Optional[float] = None, duration: Optional[float] = None) -> bool:
        return self._turn(self._rotation.dir_left, angle if angle is not None else self._rotation.angle_left,
                          duration)

    def turn_right(self, angle: Optional[float] = None, duration: Optional[float] = None) -> bool:
        return self._turn(self._rotation.dir_right, angle if angle is not None else self._rotation.angle_right,
                          duration)

    def is_busy(self) -> bool:
        return time.time() < self._busy_until

    # ------------- Internal helpers -------------
    def _move(self, direction_code: int, speed: Optional[float], duration: Optional[float]) -> bool:
        with self._lock:
            if self.is_busy():
                return False

            spd = self._movement.speed if speed is None else float(speed)
            dur = self._movement.duration if duration is None else float(duration)

            # Apply the action via UnrealCV wrapper (speed, duration, direction)
            self._ucv.apply_action_transition(self.actor_name, [spd, dur, direction_code])

            # Cooldown to avoid conflicts
            self._busy_until = time.time() + (dur * 2.0)
            return True

    def _turn(self, direction_code: int, angle: float, duration: Optional[float]) -> bool:
        with self._lock:
            if self.is_busy():
                return False

            dur = self._rotation.duration if duration is None else float(duration)
            ang = float(angle)

            # Apply the rotation via UnrealCV wrapper (duration, angle, direction)
            self._ucv.apply_action_rotation(self.actor_name, [dur, ang, direction_code])

            # Cooldown to avoid conflicts
            self._busy_until = time.time() + (dur * 2.0)
            return True

    @staticmethod
    def _load_defaults(cfg_path: str) -> tuple[MovementDefaults, RotationDefaults]:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"action_config.json not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)

        mv = data.get("action_parameters", {}).get("movement", {})
        rot = data.get("action_parameters", {}).get("rotation", {})

        # Movement
        mv_speed = float(mv.get("speed", 5000))
        mv_duration = float(mv.get("duration", 0.1))
        mv_dirs = mv.get("directions", {})
        dir_forward = int(mv_dirs.get("MOVE_FORWARD", {}).get("direction", 0))
        dir_backward = int(mv_dirs.get("MOVE_BACKWARD", {}).get("direction", 1))
        dir_left = int(mv_dirs.get("MOVE_LEFT", {}).get("direction", 2))
        dir_right = int(mv_dirs.get("MOVE_RIGHT", {}).get("direction", 3))

        movement = MovementDefaults(
            speed=mv_speed,
            duration=mv_duration,
            dir_forward=dir_forward,
            dir_backward=dir_backward,
            dir_left=dir_left,
            dir_right=dir_right,
        )

        # Rotation
        rot_duration = float(rot.get("duration", 0.1))
        rot_angle = rot.get("angle", {})
        angle_right = float(rot_angle.get("TURN_RIGHT", 90))
        angle_left = float(rot_angle.get("TURN_LEFT", -90))
        rot_dirs = rot.get("directions", {})
        dir_right = int(rot_dirs.get("TURN_RIGHT", {}).get("direction", 1))
        dir_left = int(rot_dirs.get("TURN_LEFT", {}).get("direction", -1))

        rotation = RotationDefaults(
            duration=rot_duration,
            angle_left=angle_left,
            angle_right=angle_right,
            dir_left=dir_left,
            dir_right=dir_right,
        )

        return movement, rotation
