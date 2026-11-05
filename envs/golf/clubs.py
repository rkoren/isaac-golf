"""Clubs scaled to ~1.6 m golfer"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ADDRESS_DIR = Path(__file__).with_name("address")
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


@dataclass(frozen=True)
class Club:
    grip: float                        # [m] head centre to the lead hand
    shaft_deg: float                   # shaft angle from vertical at address
    head_z: float                      # [m] head centre height at address
    hand_x: float                      # [m] lead hand in front of the ankles
    stance: float                      # [m] ankle to ankle
    head_size: tuple[float, float, float]  # [m] collider: face normal, toe-heel, height
    mass: float                        # [kg]
    loft_deg: float                    # visual for now; launch angle comes with Phase 2
    smash_max: float                   # ball speed / head speed, dead centre
    look: str                          # "driver" | "iron"
    tee: bool


CLUBS = {
    "driver": Club(0.85, 42.0, 0.089, 0.25, 0.40, (0.04, 0.10, 0.05), 0.31, 10.5, 1.47, "driver", True),
    "7iron": Club(0.69, 34.0, 0.025, 0.24, 0.34, (0.02, 0.065, 0.04), 0.40, 34.0, 1.33, "iron", True),
    "pw": Club(0.66, 32.0, 0.025, 0.23, 0.30, (0.02, 0.065, 0.042), 0.42, 46.0, 1.24, "iron", True),
}


WRIST = "lead_wrist" 


def golfer_usd(club: str, wrist: bool = False) -> str:
    return str(ASSETS_DIR / f"golfer_{club}{'_wrist' if wrist else ''}.usda")


def address(club: str) -> dict:
    """Solved address (scripts/solve_address.py): root pose, joints, ball, weld."""
    path = ADDRESS_DIR / f"{club}.json"
    if not path.exists():  # unsolved club: start from the driver's (the solver's first pass)
        print(f"[WARN] no address for {club}, using the driver's")
        path = ADDRESS_DIR / "driver.json"
    return json.loads(path.read_text())
