"""Build assets/: vendor the stock humanoid as ascii usd, then one golfer per club (club welded to the left hand).

Vendoring keeps the asset offline-reproducible (the upstream lives on a CDN and the local
copy sits in %TEMP%, which cleaners eat). Flattened to .usda so it stays diffable in review.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from envs.golf.clubs import CLUBS, WRIST, Club, address, golfer_usd

SHAFT_R = 0.015  # collider
GROUND_SHAFT_Z0 = 0.06  # [m] ball on the ground: shaft collider starts above the hosel, clear of a divot
HEAD_SHARE = 0.62       # ball on the ground: head's share of the club mass, as a real iron (~265 of ~430 g)
WRIST_LIMITS = (-25.0, 70.0)  # [deg] from the address angle
# full arms (--arms full)
TWIST_LIMIT = 75.0            # [deg] upper-arm twist each way
ELBOW_STRAIGHT = -109.0       # [deg] stock -90 leaves 19.5 deg of bend; -109.47 is straight
WRIST_AT = 0.77               # wrist pivot along elbow -> hand centre (0.24 of 0.31 m, as a person's wrist-to-grip)
LEAD_COCK = (-30.0, 80.0)     # [deg] from address, about the club's toe-heel axis
LEAD_BOW = (-45.0, 45.0)      # [deg]
TRAIL_WRIST = (-60.0, 60.0)   # [deg] both axes
REPO = Path(__file__).resolve().parents[1]
UPSTREAM = "{ISAAC}/Robots/IsaacSim/Humanoid/humanoid.usd"


def vendor_humanoid(out: Path) -> None:
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

    src = retrieve_file_path(UPSTREAM.format(ISAAC=ISAAC_NUCLEUS_DIR))
    stage = Usd.Stage.Open(str(src))
    stage.Flatten().Export(str(out))


def material(stage, path: str, color, roughness: float, metallic: float = 0.0) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def mesh(stage, path: str, verts, faces, mat, smooth: bool = True) -> None:
    """Visual mesh; smooth vertex normals, or faceted for flat-sided parts."""
    v = np.asarray(verts, dtype=float)
    fn = [sum(np.cross(v[f][i] - v[f][0], v[f][i + 1] - v[f][0]) for i in range(1, len(f) - 1)) for f in faces]
    if smooth:
        n = np.zeros_like(v)
        for f, x in zip(faces, fn):
            n[f] += x
    else:
        n = np.array([x for f, x in zip(faces, fn) for _ in f])
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in v])
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
    m.CreateNormalsAttr([Gf.Vec3f(*x) for x in n])
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex if smooth else UsdGeom.Tokens.faceVarying)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


def rod(stage, path: str, p0, p1, r: float, mat) -> None:
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    d = p1 - p0
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(r)
    cyl.CreateHeightAttr(float(np.linalg.norm(d)))
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    q = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(d / np.linalg.norm(d)))).GetQuat()
    cyl.AddTranslateOp().Set(Gf.Vec3d(*((p0 + p1) / 2)))
    cyl.AddOrientOp().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mat)


def loft_rings(rings: list[np.ndarray], apex=None):
    """Stack of closed CCW rings (bottom to top) -> verts, faces; bottom capped, top capped or apexed."""
    k, m = len(rings), len(rings[0])
    verts = [p for r in rings for p in r]
    faces = [[a * m + i, a * m + (i + 1) % m, (a + 1) * m + (i + 1) % m, (a + 1) * m + i]
             for a in range(k - 1) for i in range(m)]
    bottom = len(verts)
    verts.append(rings[0].mean(0))
    faces += [[bottom, (i + 1) % m, i] for i in range(m)]
    top = len(verts)
    verts.append(rings[-1].mean(0) if apex is None else np.asarray(apex))
    faces += [[top, (k - 1) * m + i, (k - 1) * m + (i + 1) % m] for i in range(m)]
    return verts, faces


def superellipse(a: float, b: float, n: float, count: int = 64) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, count, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([np.sign(c) * np.abs(c) ** (2 / n) * a, np.sign(s) * np.abs(s) ** (2 / n) * b], 1)


def driver_look(xf: float):
    """460cc-style pear in head coords (x toward target, y toward heel, z up), flat face at x = xf."""
    xc, sole, crown = xf - 0.040, -0.025, 0.033
    outline = superellipse(0.060, 0.058, 2.4)
    levels = [(sole, 0.90, 0.0), (sole + 0.004, 0.98, 0.0), (sole + 0.012, 1.0, 0.0)]
    levels += [(crown * f, math.sqrt(1 - f * f), -0.012 * f) for f in np.linspace(0.0, 0.95, 8)]
    rings = []
    for z, k, dx in levels:
        r = outline * k
        rings.append(np.stack([np.minimum(xc + dx + r[:, 0], xf), r[:, 1], np.full(len(r), z)], 1))
    body = loft_rings(rings, apex=(xc - 0.012, 0.0, crown))
    # face plate just proud of the flat clip
    zlo, zhi = sole + 0.003, 0.026
    plate = superellipse(0.046, (zhi - zlo) / 2, 4.0)
    pts = [np.array([xf + 0.0008, y, (zlo + zhi) / 2 + z]) for y, z in plate]
    face = (pts + [np.array([xf + 0.0008, 0.0, (zlo + zhi) / 2])],
            [[len(pts), i, (i + 1) % len(pts)] for i in range(len(pts))])
    hosel = np.array([xf - 0.012, 0.046, 0.018])
    return body, face, hosel


def iron_look(xf: float, loft_deg: float):
    """Cavity-back blade in head coords, leaning back by its loft about the leading edge."""
    sole, heel, toe = -0.020, 0.032, -0.033
    outline = [(heel, sole), (heel, 0.010)]
    outline += [(heel + (toe + 0.010 - heel) * f, 0.010 + 0.010 * f) for f in np.linspace(0.1, 1.0, 8)]
    outline += [(toe + 0.010 + 0.010 * math.cos(a), 0.010 + 0.010 * math.sin(a)) for a in np.linspace(math.pi / 2, math.pi, 6)]
    outline += [(toe + 0.006 * math.cos(a), sole + 0.006 + 0.018 * math.sin(a)) for a in np.linspace(math.pi, 1.5 * math.pi, 6)]
    yz = np.array(outline)
    if np.sum(yz[:, 0] * np.roll(yz[:, 1], -1) - np.roll(yz[:, 0], -1) * yz[:, 1]) < 0:
        yz = yz[::-1]  # ccw in (y, z) so the face points +x
    back = xf - (0.012 + 0.010 * (0.022 - yz[:, 1]) / 0.042)  # thicker at the sole
    front = np.stack([np.full(len(yz), xf), yz[:, 0], yz[:, 1]], 1)
    rear = np.stack([back, yz[:, 0], yz[:, 1]], 1)
    m = len(yz)
    verts = list(front) + list(rear) + [front.mean(0), rear.mean(0)]
    faces = [[2 * m, i, (i + 1) % m] for i in range(m)]
    faces += [[2 * m + 1, m + (i + 1) % m, m + i] for i in range(m)]
    faces += [[i, m + i, m + (i + 1) % m, (i + 1) % m] for i in range(m)]
    c, s = math.cos(math.radians(loft_deg)), math.sin(math.radians(loft_deg))
    lean = lambda p: np.array([xf + c * (p[0] - xf) - s * (p[2] - sole), p[1], sole + s * (p[0] - xf) + c * (p[2] - sole)])
    return ([lean(p) for p in verts], faces), lean(np.array([xf - 0.008, heel - 0.002, 0.008]))


def club_mass_props(club, spec: Club) -> None:
    """Authored mass properties: head box at the origin plus a thin shaft rod up to the grip."""
    m = spec.mass
    mh, mr = HEAD_SHARE * m, (1.0 - HEAD_SHARE) * m
    a, b, c = spec.head_size
    length = spec.grip
    zc = mr * length / 2 / m
    rod = mr * length**2 / 12 + mr * (length / 2 - zc) ** 2 + mh * zc**2
    api = UsdPhysics.MassAPI(club)
    api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, zc))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(mh * (b * b + c * c) / 12 + rod, mh * (a * a + c * c) / 12 + rod,
                                           mh * (a * a + b * b) / 12 + mr * SHAFT_R**2 / 2))
    api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def d6_joint(stage, path: str, body0: str, body1: str, pos0, pos1, rot, limits: dict[str, tuple[float, float]]):
    """Joint with free rotation axes (limits [deg]) and every other axis locked; rot = joint frame in both bodies."""
    j = UsdPhysics.Joint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    j.CreateLocalRot0Attr(rot)
    j.CreateLocalRot1Attr(rot)
    for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        lo, hi = limits.get(axis, (1.0, -1.0))  # low > high locks the axis
        lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
        lim.CreateLowAttr(lo)
        lim.CreateHighAttr(hi)
        if axis in limits:
            UsdPhysics.DriveAPI.Apply(j.GetPrim(), axis).CreateTypeAttr("force")  # gains from the actuator cfg
    return j


def frame_quat(x, y) -> Gf.Quatf:
    """Rotation whose X axis is x and Y axis is y projected off x (USD w, x, y, z)."""
    x = np.asarray(x, float) / np.linalg.norm(x)
    y = np.asarray(y, float) - np.dot(y, x) * x
    y /= np.linalg.norm(y)
    m = Gf.Matrix3d(*np.stack([x, y, np.cross(x, y)]).flatten().tolist())  # rows: row-vector convention
    q = m.ExtractRotation().GetQuat()
    return Gf.Quatf(q.GetReal(), *q.GetImaginary())


def full_arms(stage, weld_rot) -> None:
    """Upper-arm twist, elbows that straighten, and a 2-axis wrist on each hand, pivoting where a person's does."""
    for side in ("left", "right"):
        sh = stage.OverridePrim(f"/golfer/joints/{side}_upper_arm")
        sh.CreateAttribute("limit:rotY:physics:low", Sdf.ValueTypeNames.Float).Set(-TWIST_LIMIT)
        sh.CreateAttribute("limit:rotY:physics:high", Sdf.ValueTypeNames.Float).Set(TWIST_LIMIT)
        UsdPhysics.DriveAPI.Apply(sh, "rotY").CreateTypeAttr("force")
        stage.OverridePrim(f"/golfer/joints/{side}_lower_arm").CreateAttribute(
            "physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(ELBOW_STRAIGHT)
        stage.OverridePrim(f"/golfer/joints/{side}_hand").SetActive(False)
        # stock hand joint offset on the forearm; the wrist pivots WRIST_AT of the way to the hand centre
        to_hand = np.array([0.18, 0.18 if side == "right" else -0.18, 0.18])
        if side == "left":
            # cock about the club's toe-heel axis at address (today's hinge), bow/cup perpendicular to it
            toe_heel = Gf.Quatf(*weld_rot).GetInverse().Transform(Gf.Vec3f(0, 1, 0))
            rot, limits = frame_quat(to_hand, list(toe_heel)), {"rotY": LEAD_COCK, "rotZ": LEAD_BOW}
        else:
            rot, limits = frame_quat(to_hand, [0.0, -1.0, 1.0]), {"rotY": TRAIL_WRIST, "rotZ": TRAIL_WRIST}
        d6_joint(stage, f"/golfer/joints/{side}_wrist", f"/golfer/{side}_lower_arm", f"/golfer/{side}_hand",
                 WRIST_AT * to_hand, (WRIST_AT - 1.0) * to_hand, rot, limits)


def build_golfer(humanoid_rel: str, out: Path, spec: Club, weld_rot, wrist: bool = False, arms: str = "stock") -> None:
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    golfer = UsdGeom.Xform.Define(stage, "/golfer").GetPrim()
    golfer.GetReferences().AddReference(humanoid_rel)
    stage.SetDefaultPrim(golfer)

    club = UsdGeom.Xform.Define(stage, "/golfer/club").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(club)
    UsdPhysics.MassAPI.Apply(club).CreateMassAttr(spec.mass)

    # colliders: the strike proxy and the shaft, hidden from render
    head = UsdGeom.Cube.Define(stage, "/golfer/club/head")
    head.CreateSizeAttr(1.0)
    head.AddScaleOp().Set(Gf.Vec3f(*spec.head_size))  # cube size is edge length
    UsdPhysics.CollisionAPI.Apply(head.GetPrim())
    z0 = 0.0
    if not spec.tee:
        # ball on the ground: the head is the strike and turf proxy only (analytic), never touching the ground
        UsdPhysics.CollisionAPI(head.GetPrim()).CreateCollisionEnabledAttr(False)
        club_mass_props(club, spec)
        z0 = GROUND_SHAFT_Z0
    shaft_len = spec.grip - 0.01 - z0
    shaft = UsdGeom.Capsule.Define(stage, "/golfer/club/shaft")
    shaft.CreateRadiusAttr(SHAFT_R)
    shaft.CreateHeightAttr(shaft_len)
    shaft.CreateAxisAttr(UsdGeom.Tokens.z)
    shaft.AddTranslateOp().Set(Gf.Vec3d(0, 0, z0 + shaft_len / 2 + SHAFT_R))
    UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())
    for g in (head, shaft):
        g.CreatePurposeAttr(UsdGeom.Tokens.guide)

    # looks: head level with the ground at address (tilted off the shaft by the lie)
    looks = "/golfer/club/Looks"
    xf = spec.head_size[0] / 2
    lie = UsdGeom.Xform.Define(stage, "/golfer/club/look")
    lie.AddRotateXOp().Set(spec.shaft_deg)
    if spec.look == "driver":
        (bv, bf), (fv, ff), hosel = driver_look(xf)
        mesh(stage, "/golfer/club/look/body", bv, bf, material(stage, f"{looks}/crown", (0.03, 0.03, 0.035), 0.35))
        mesh(stage, "/golfer/club/look/face", fv, ff, material(stage, f"{looks}/face", (0.55, 0.56, 0.58), 0.3, 1.0))
    else:
        (bv, bf), hosel = iron_look(xf, spec.loft_deg)
        mesh(stage, "/golfer/club/look/body", bv, bf, material(stage, f"{looks}/chrome", (0.75, 0.76, 0.78), 0.2, 1.0), False)
    t = math.radians(spec.shaft_deg)
    tilt = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    base, grip_end = tilt @ hosel, np.array([0.0, 0.0, spec.grip + 0.005])
    up = (grip_end - base) / np.linalg.norm(grip_end - base)
    steel = material(stage, f"{looks}/steel", (0.62, 0.63, 0.65), 0.3, 1.0)
    rod(stage, "/golfer/club/hosel", base, base + 0.04 * up, 0.0075, steel)
    rod(stage, "/golfer/club/shaft_look", base, grip_end, 0.0055, steel)
    rod(stage, "/golfer/club/grip_look", grip_end - 0.18 * up, grip_end, 0.0105,
        material(stage, f"{looks}/rubber", (0.04, 0.04, 0.04), 0.8))

    if arms == "full":
        full_arms(stage, weld_rot)
    if wrist and arms != "full":
        # hinge about the club's toe-heel axis: the swing plane at address (shaft + face normal)
        weld = UsdPhysics.RevoluteJoint.Define(stage, f"/golfer/{WRIST}")
        weld.CreateAxisAttr(UsdPhysics.Tokens.y)
        weld.CreateLowerLimitAttr(WRIST_LIMITS[0])
        weld.CreateUpperLimitAttr(WRIST_LIMITS[1])
        UsdPhysics.DriveAPI.Apply(weld.GetPrim(), "angular")  # gains come from the actuator cfg
        rot0, rot1 = Gf.Quatf(*weld_rot).GetInverse(), Gf.Quatf(1, 0, 0, 0)  # joint frame = club frame
    else:
        weld = UsdPhysics.FixedJoint.Define(stage, "/golfer/club_weld")
        rot0, rot1 = Gf.Quatf(1, 0, 0, 0), Gf.Quatf(*weld_rot)
    weld.CreateBody0Rel().SetTargets([Sdf.Path("/golfer/left_hand")])
    weld.CreateBody1Rel().SetTargets([Sdf.Path("/golfer/club")])
    weld.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    weld.CreateLocalPos1Attr(Gf.Vec3f(0, 0, spec.grip))
    weld.CreateLocalRot0Attr(rot0)
    weld.CreateLocalRot1Attr(rot1)

    # the grip cluster never collides with itself
    grip = ["club", "left_hand", "right_hand", "left_lower_arm", "right_lower_arm"]
    for i, body in enumerate(grip[:-1]):
        prim = club if body == "club" else stage.OverridePrim(f"/golfer/{body}")
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(
            [Sdf.Path(f"/golfer/{other}") for other in grip[i + 1:]]
        )

    stage.GetRootLayer().Save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default=str(REPO / "assets"))
    ap.add_argument("--club", default="all", help=f"one of {list(CLUBS)} or all")
    ap.add_argument("--revendor", action="store_true", help="refetch the humanoid even if vendored")
    ap.add_argument("--wrist", action="store_true", help="hinge the club to the lead hand instead of welding it")
    ap.add_argument("--arms", default="stock", choices=("stock", "full"),
                    help="full: upper-arm twist, straightening elbows, 2-axis wrists on both hands (club welded)")
    a = ap.parse_args()

    assets = Path(a.assets)
    assets.mkdir(parents=True, exist_ok=True)
    humanoid = assets / "humanoid.usda"
    if a.revendor or not humanoid.exists():
        vendor_humanoid(humanoid)
        print(f"vendored {humanoid}")

    for name in CLUBS if a.club == "all" else [a.club]:
        # weld from the club's solved address; before its first solve, borrow the driver's
        weld_rot = address(name, a.arms)["weld_rot"]
        out = Path(golfer_usd(name, a.wrist, a.arms))
        if a.assets != str(REPO / "assets"):
            out = assets / out.name
        out.unlink(missing_ok=True)
        build_golfer("./humanoid.usda", out, CLUBS[name], weld_rot, a.wrist, a.arms)
        print(f"wrote {out}")
