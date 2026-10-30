"""Golfer collision shapes from its USD (pxr only)"""

from __future__ import annotations


def body_shapes(usd_path: str) -> dict[str, list[tuple]]:
    """Shapes per body in body frame: ("cap", a, b, r) | ("sph", c, r) | ("box", c, c+hx, c+hy, c+hz)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    shapes: dict[str, list[tuple]] = {}
    seen = set()
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            continue
        # geom -> body. usd matrices are row-vector, hence geom * inverse(body)
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0) * (
            UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0).GetInverse()
        )
        p = lambda *v: tuple(round(c, 5) for c in m.Transform(Gf.Vec3d(*v)))
        kind = prim.GetTypeName()
        if kind == "Capsule":
            g = UsdGeom.Capsule(prim)
            h, r = g.GetHeightAttr().Get() / 2, g.GetRadiusAttr().Get()
            axis = [0.0, 0.0, 0.0]
            axis["XYZ".index(g.GetAxisAttr().Get())] = h
            shape = ("cap", p(*axis), p(*(-a for a in axis)), round(r, 4))
        elif kind == "Sphere":
            shape = ("sph", p(0, 0, 0), round(UsdGeom.Sphere(prim).GetRadiusAttr().Get(), 4))
        elif kind == "Cube":
            s = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            shape = ("box", p(0, 0, 0), p(s, 0, 0), p(0, s, 0), p(0, 0, s))
        else:
            continue
        key = (body.GetName(), shape)
        if key not in seen:  # vendored asset authors some shapes twice
            seen.add(key)
            shapes.setdefault(body.GetName(), []).append(shape)
    return shapes


def capsule_table(usd_path: str, body_names: list[str]):
    """Every shape as a capsule (box ~ capsule on its long axis): local points, owners, (a, b, r, body)."""
    import numpy as np

    shapes = body_shapes(usd_path)
    local, owner, caps = [], [], []

    def add(b: int, pt) -> int:
        local.append(tuple(pt))
        owner.append(b)
        return len(local) - 1

    for b, name in enumerate(body_names):
        for sh in shapes.get(name, []):
            if sh[0] == "cap":
                caps.append((add(b, sh[1]), add(b, sh[2]), sh[3], name))
            elif sh[0] == "sph":
                i = add(b, sh[1])
                caps.append((i, i, sh[2], name))
            else:
                c = np.array(sh[1])
                half = sorted((np.array(sh[k]) - c for k in (2, 3, 4)), key=np.linalg.norm)
                caps.append((add(b, c - half[2]), add(b, c + half[2]), float(np.linalg.norm(half[1])), name))
    return local, owner, caps


def collision_pairs(usd_path: str, caps: list[tuple]) -> list[tuple[int, int]]:
    """Capsule index pairs physx will actually collide: different bodies, not skipped."""
    skip = skipped_pairs(usd_path)
    return [
        (i, j)
        for i in range(len(caps))
        for j in range(i + 1, len(caps))
        if caps[i][3] != caps[j][3] and frozenset((caps[i][3], caps[j][3])) not in skip
    ]


def seg_dist(p1, q1, p2, q2):
    """Batched closest distance between segments p1q1 and p2q2 (Ericson 5.1.9)"""
    import torch

    eps = 1e-9
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    den = a * e - b * b
    s = torch.where(den > eps, ((b * f - c * e) / den.clamp_min(eps)).clamp(0, 1), torch.zeros_like(a))
    t = (b * s + f) / e.clamp_min(eps)
    s = torch.where(t < 0, (-c / a.clamp_min(eps)).clamp(0, 1), torch.where(t > 1, ((b - c) / a.clamp_min(eps)).clamp(0, 1), s))
    t = t.clamp(0, 1)
    # zero-length segments
    s = torch.where(a <= eps, torch.zeros_like(s), s)
    t = torch.where(a <= eps, (f / e.clamp_min(eps)).clamp(0, 1), t)
    s = torch.where((e <= eps) & (a > eps), (-c / a.clamp_min(eps)).clamp(0, 1), s)
    t = torch.where(e <= eps, torch.zeros_like(t), t)
    return torch.linalg.norm((p1 + d1 * s.unsqueeze(-1)) - (p2 + d2 * t.unsqueeze(-1)), dim=-1)


def penetrations(world, caps, pairs):
    """Penetration depth [m] per pair, (..., P); positive = overlapping. world: (..., K, 3)"""
    import torch

    i = torch.tensor([caps[p][0] for p, _ in pairs], device=world.device)
    j = torch.tensor([caps[p][1] for p, _ in pairs], device=world.device)
    k = torch.tensor([caps[q][0] for _, q in pairs], device=world.device)
    m = torch.tensor([caps[q][1] for _, q in pairs], device=world.device)
    radii = torch.tensor([caps[p][2] + caps[q][2] for p, q in pairs], device=world.device)
    return radii - seg_dist(world[..., i, :], world[..., j, :], world[..., k, :], world[..., m, :])


def skipped_pairs(usd_path: str) -> set[frozenset[str]]:
    """Body pairs physx never collides: joined by a joint, or listed in a FilteredPairsAPI."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    pairs = set()
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            b = [t.name for rel in (j.GetBody0Rel(), j.GetBody1Rel()) for t in rel.GetTargets()]
            if len(b) == 2:
                pairs.add(frozenset(b))
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            for t in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                pairs.add(frozenset((prim.GetName(), t.name)))
    return pairs
