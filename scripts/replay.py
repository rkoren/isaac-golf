"""The settings that replay a run's own body and club, from its env.yaml (reward-only settings don't change the swing)"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPLAY = ("club", "wrist", "muscle", "muscle_k", "damping_scale", "leg_damping_scale", "impact_arm_dist",
          "max_depenetration_velocity", "motor_noise", "motor_noise_tau", "ball_forward", "wrist_gear", "flight_obs",
          "turf_area", "ankle_gear", "grip_spring", "grip_damping", "grip_max_force", "gravity_obs", "ball_ref_obs",
          "arms")


def replay_overrides(r: Path) -> list[str]:
    env_yaml = (r / "params" / "env.yaml").read_text(encoding="utf-8") if (r / "params" / "env.yaml").exists() else ""
    ov = [f"env.{k}={m.group(1)}" for k in REPLAY if (m := re.search(rf"^{k}: (\S+)", env_yaml, re.M))]
    # runs before the knob only have the robot's nested value
    if not any("max_depenetration_velocity" in o for o in ov) and (
        m := re.search(r"^\s+max_depenetration_velocity: ([\d.]+)", env_yaml, re.M)
    ):
        ov.append(f"env.max_depenetration_velocity={m.group(1)}")
    return ov


def viewer_data(html: Path) -> dict:
    """the swing data a record_swings.py viewer embeds"""
    blob = re.search(r'id="swing-data" type="application/json">(.*?)</script>', html.read_text(encoding="utf-8"), re.S).group(1)
    return json.loads(blob.replace("<\\/", "</"))
