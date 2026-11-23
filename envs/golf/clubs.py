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
    loft_deg: float                    # loft for launch angle and spin (flight/)
    smash_max: float                   # ball speed / head speed, dead centre
    look: str                          # "driver" | "iron"
    tee: bool                          # False: the ball rests on the ground (analytic turf)
    trackman: str                      # row of flight/data.py whose spin rate this club's is calibrated on


CLUBS = {
    "driver": Club(0.85, 42.0, 0.089, 0.25, 0.40, (0.04, 0.10, 0.05), 0.31, 10.5, 1.47, "driver", True, "Driver"),
    "7iron": Club(0.69, 34.0, 0.0213, 0.24, 0.34, (0.02, 0.065, 0.04), 0.40, 34.0, 1.33, "iron", False, "7 Iron"),
    "pw": Club(0.66, 32.0, 0.025, 0.23, 0.30, (0.02, 0.065, 0.042), 0.42, 46.0, 1.24, "iron", True, "PW"),
}
CLUBS["7iron_arm"] = CLUBS["7iron"]  # trial: the 7-iron addressed with a straight lead arm
CLUBS["driver_arm"] = CLUBS["driver"]  # trial: the driver addressed the same way


WRIST = "lead_wrist" 


def golfer_usd(club: str, wrist: bool = False, arms: str = "stock") -> str:
    suffix = "_full" if arms == "full" else "_wrist" if wrist else ""
    return str(ASSETS_DIR / f"golfer_{club}{suffix}.usda")


def address_path(club: str, arms: str = "stock") -> Path:
    return ADDRESS_DIR / f"{club}{'_full' if arms == 'full' else ''}.json"


def address(club: str, arms: str = "stock") -> dict:
    """Solved address (scripts/solve_address.py): root pose, joints, ball, weld."""
    path = address_path(club, arms)
    if not path.exists():  # unsolved: the stock body's, then the driver's (the solver's first pass)
        fallback = address_path(club) if address_path(club).exists() else ADDRESS_DIR / "driver.json"
        print(f"[WARN] no address {path.name}, using {fallback.name}")
        path = fallback
    return json.loads(path.read_text())
