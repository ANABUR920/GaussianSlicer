"""Prepare binary parallel contours without a ground-truth mesh.

JSON schema: [{"origin": [x,y,z], "normal": [nx,ny,nz],
               "loops": [[[x,y,z], ...], ...]}, ...].
Nested loops use the even-odd filling rule. CSLC polylines are also accepted.
"""

import argparse, json
from pathlib import Path
import numpy as np
from shapely import contains_xy, distance, points
from gaussian_slicer import make_regions


def read_csl(path):
    lines = iter(x.strip() for x in Path(path).read_text().splitlines() if x.strip())
    if next(lines) != "CSLC":
        raise ValueError("Expected CSLC format")
    count, _ = map(int, next(lines).split())
    planes = []
    for _ in range(count):
        h = next(lines).split()
        nv, nc = int(h[1]), int(h[2])
        eq = np.asarray(h[3:7], float)
        vertices = np.asarray(
            [list(map(float, next(lines).split())) for _ in range(nv)]
        )
        loops = []
        for _ in range(nc):
            fields = next(lines).split()
            n = int(fields[0])
            ids = np.asarray(fields[-n:], int)
            loops.append(vertices[ids].tolist())
        if loops:
            planes.append(
                dict(normal=eq[:3].tolist(), origin=vertices[0].tolist(), loops=loops)
            )
    return planes


def prepare(planes, out, normalize=True, seed=11):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vertices = np.concatenate(
        [np.asarray(l, float) for p in planes for l in p["loops"]]
    )
    if not np.isfinite(vertices).all():
        raise ValueError("Non-finite contour coordinates")
    center = (vertices.min(0) + vertices.max(0)) / 2 if normalize else np.zeros(3)
    scale = 1.8 / np.ptp(vertices, axis=0).max() if normalize else 1.0
    clean = []
    for p in planes:
        n = np.asarray(p["normal"], float)
        n /= np.linalg.norm(n)
        o = (np.asarray(p["origin"]) - center) * scale
        loops = []
        for loop in p["loops"]:
            a = (np.asarray(loop, float) - center) * scale
            if np.linalg.norm(a[-1] - a[0]) < 1e-9:
                a = a[:-1]
            if len(a) < 3:
                raise ValueError("A contour needs at least three distinct vertices")
            if np.max(abs((a - o) @ n)) > 1e-5:
                raise ValueError("Contour does not lie in its plane")
            loops.append(a.tolist())
        clean.append(dict(origin=o.tolist(), normal=n.tolist(), loops=loops))
    regions, frames = make_regions(clean)
    normals = np.asarray([f[2] for f in frames])
    positions = np.asarray([f[0] @ normals[0] for f in frames])
    if len(clean) < 2 or np.min(abs(normals @ normals[0])) < 0.9999:
        raise ValueError("At least two parallel planes required")
    if np.diff(np.sort(positions)).min() < 1e-6:
        raise ValueError("Duplicate plane locations")
    on = []
    off = []
    sdf = []
    pid = []
    for i, (p, r, (o, B, n)) in enumerate(zip(clean, regions, frames)):
        if r.is_empty:
            raise ValueError("Empty filled region")
        candidates = []
        edges = []
        for loop in p["loops"]:
            a = (np.asarray(loop) - o) @ B
            d = np.roll(a, -1, axis=0) - a
            valid = np.linalg.norm(d, axis=1) > 1e-10
            a = a[valid]
            d = d[valid]
            q = (
                a[:, None] + d[:, None] * ((np.arange(4) + 0.5) / 4)[None, :, None]
            ).reshape(-1, 2)
            edges.append(q)
            normal = np.repeat(
                np.c_[d[:, 1], -d[:, 0]] / np.linalg.norm(d, axis=1)[:, None], 4, axis=0
            )
            for eps in [1 / 32, 1 / 64, 1 / 128]:
                candidates.extend([q + eps * normal, q - eps * normal])
        lo = np.asarray(r.bounds[:2]) - 0.15
        hi = np.asarray(r.bounds[2:]) + 0.15
        candidates.append(rng.uniform(lo, hi, (2000, 2)))
        for component in ([r] if r.geom_type == "Polygon" else r.geoms):
            q = rng.uniform(component.bounds[:2], component.bounds[2:], (4000, 2))
            candidates.append(q[contains_xy(component, q[:, 0], q[:, 1])][:100])
        x = np.concatenate(candidates)
        sd = distance(r.boundary, points(x))
        sd[contains_xy(r, x[:, 0], x[:, 1])] *= -1
        on.append(np.concatenate(edges) @ B.T + o)
        off.append(x @ B.T + o)
        sdf.append(sd)
        pid.append(np.full(len(x), i))
    (out / "planes.json").write_text(json.dumps(clean))
    np.savez_compressed(
        out / "samples.npz",
        on=np.concatenate(on).astype("f"),
        off=np.concatenate(off).astype("f"),
        sdf=np.concatenate(sdf).astype("f"),
        plane_id=np.concatenate(pid),
    )
    (out / "transform.json").write_text(
        json.dumps(
            dict(
                center=center.tolist(),
                scale=scale,
                forward="normalized=(world-center)*scale",
                inverse="world=normalized/scale+center",
                seed=seed,
            ),
            indent=2,
        )
    )
    return clean


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--keep-coordinates", action="store_true")
    a = ap.parse_args()
    source = Path(a.input)
    planes = (
        read_csl(source)
        if source.suffix.lower() == ".csl"
        else json.loads(source.read_text())
    )
    prepare(planes, a.output, not a.keep_coordinates)
