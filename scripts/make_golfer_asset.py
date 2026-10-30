"""Build assets/: vendor the stock humanoid as ascii usd, then weld a club to its left hand.

Vendoring keeps the asset offline-reproducible (the upstream lives on a CDN and the local
copy sits in %TEMP%, which cleaners eat). Flattened to .usda so it stays diffable in review.
"""

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# size club for golfer (currently driver)
SHAFT_R = 0.015
SHAFT_LEN = 0.84 # cylinder
HEAD_SIZE = (0.04, 0.10, 0.05)
CLUB_MASS = 0.31
GRIP_Z = 0.85 # where hand grabs
# club orientation in hand
WELD_ROT = (0.00377, 0.79605, 0.25261, -0.54958) # (w,x,y,z)

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = "{ISAAC}/Robots/IsaacSim/Humanoid/humanoid.usd"


def vendor_humanoid(out: Path) -> None:
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

    src = retrieve_file_path(UPSTREAM.format(ISAAC=ISAAC_NUCLEUS_DIR))
    stage = Usd.Stage.Open(str(src))
    stage.Flatten().Export(str(out))


def build_golfer(humanoid_rel: str, out: Path) -> None:
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    golfer = UsdGeom.Xform.Define(stage, "/golfer").GetPrim()
    golfer.GetReferences().AddReference(humanoid_rel)
    stage.SetDefaultPrim(golfer)

    club = UsdGeom.Xform.Define(stage, "/golfer/club").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(club)
    UsdPhysics.MassAPI.Apply(club).CreateMassAttr(CLUB_MASS)

    head = UsdGeom.Cube.Define(stage, "/golfer/club/head")
    head.CreateSizeAttr(1.0)
    head.AddScaleOp().Set(Gf.Vec3f(*[s / 2.0 for s in HEAD_SIZE]))  # size=1 cube spans -1..1
    UsdPhysics.CollisionAPI.Apply(head.GetPrim())

    shaft = UsdGeom.Capsule.Define(stage, "/golfer/club/shaft")
    shaft.CreateRadiusAttr(SHAFT_R)
    shaft.CreateHeightAttr(SHAFT_LEN)
    shaft.CreateAxisAttr(UsdGeom.Tokens.z)
    shaft.AddTranslateOp().Set(Gf.Vec3d(0, 0, SHAFT_LEN / 2 + SHAFT_R))
    UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())

    weld = UsdPhysics.FixedJoint.Define(stage, "/golfer/club_weld")
    weld.CreateBody0Rel().SetTargets([Sdf.Path("/golfer/left_hand")])
    weld.CreateBody1Rel().SetTargets([Sdf.Path("/golfer/club")])
    weld.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    weld.CreateLocalPos1Attr(Gf.Vec3f(0, 0, GRIP_Z))
    weld.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    weld.CreateLocalRot1Attr(Gf.Quatf(*WELD_ROT))

    stage.GetRootLayer().Save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default=str(REPO / "assets"))
    ap.add_argument("--revendor", action="store_true", help="refetch the humanoid even if vendored")
    a = ap.parse_args()

    assets = Path(a.assets)
    assets.mkdir(parents=True, exist_ok=True)
    humanoid = assets / "humanoid.usda"
    golfer = assets / "golfer.usda"

    if a.revendor or not humanoid.exists():
        vendor_humanoid(humanoid)
        print(f"vendored {humanoid}")

    if golfer.exists():
        golfer.unlink()
    build_golfer("./humanoid.usda", golfer)
    print(f"wrote {golfer}")
