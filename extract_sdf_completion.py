"""Construct a surface from learned Gaussian occupancy masks and slice spacing."""

import argparse, json, time
from pathlib import Path
import numpy as np, torch, trimesh
from scipy.ndimage import distance_transform_edt, label, maximum, gaussian_filter
from scipy.interpolate import CubicSpline
from skimage.measure import marching_cubes
from gaussian_slicer import PlanarGaussians, make_regions


def build(planes, run, out, res, device, interpolation="linear", smooth_sigma=0.0):
    start = time.time()
    meta = json.loads((run / "metadata.json").read_text())
    r, f = make_regions(planes)
    m = PlanarGaussians(planes, r, f, meta["kernels_per_plane"], 11).to(device)
    m.load_state_dict(
        torch.load(run / "model.pt", weights_only=True, map_location=device)
    )
    n = m.normal[0].cpu().numpy()
    axis_id = int(np.argmax(abs(n)))
    assert abs(n[axis_id]) > 1 - 1e-6
    axes = [i for i in range(3) if i != axis_id]
    axis = np.linspace(-1.1, 1.1, res, dtype=np.float32)
    dx = 2.2 / (res - 1)
    flat = np.arange(res * res)
    points = np.zeros((res * res, 3), np.float32)
    points[:, axes[0]] = axis[flat // res]
    points[:, axes[1]] = axis[flat % res]
    fields = []
    radii = []
    with torch.no_grad():
        for pi in range(len(planes)):
            vals = []
            for start_i in range(0, len(points), 4096):
                x = torch.from_numpy(points[start_i : start_i + 4096]).to(device)
                idx = torch.full((len(x),), pi, dtype=torch.long, device=device)
                vals.append(m.plane_occupancy(x, idx).cpu().numpy())
            mask = np.concatenate(vals).reshape(res, res) >= 0.5
            inside = distance_transform_edt(mask) * dx
            outside = distance_transform_edt(~mask) * dx
            fields.append((outside - inside).astype("f"))
            labels, count = label(mask)
            peak = np.r_[dx, maximum(inside, labels, index=np.arange(1, count + 1))]
            radii.append(peak[labels].astype("f"))
    coordinate = m.coordinate.cpu().numpy()
    spacing = m.spacing.cpu().numpy()
    order = coordinate.argsort()
    s = coordinate[order]
    spline = (
        CubicSpline(s, np.asarray(fields)[order], axis=0, bc_type="natural")
        if interpolation == "cubic"
        else None
    )
    volume = np.empty((res, res, res), np.float32)
    for i, z in enumerate(axis * n[axis_id]):
        if s[0] <= z <= s[-1]:
            right = np.clip(np.searchsorted(s, z), 1, len(s) - 1)
            left = right - 1
            t = (z - s[left]) / (s[right] - s[left])
            v = (
                spline(z)
                if spline is not None
                else (1 - t) * fields[order[left]] + t * fields[order[right]]
            )
        else:
            endpoint = order[0] if z < s[0] else order[-1]
            outside = abs(z - coordinate[endpoint])
            sd = fields[endpoint]
            radius = np.maximum(radii[endpoint], dx)
            height = np.minimum(radius, 0.5 * spacing[endpoint])
            v = (
                np.sqrt(
                    np.maximum(sd + radius, 0) ** 2 + (outside * radius / height) ** 2
                )
                - radius
            )
        index = [slice(None)] * 3
        index[axis_id] = i
        volume[tuple(index)] = v
    if smooth_sigma > 0:
        # Sigma is in normalized world coordinates, independent of grid resolution.
        volume = gaussian_filter(volume, smooth_sigma / dx, mode="nearest")
    vertices, faces, _, _ = marching_cubes(
        volume, 0, spacing=(dx,) * 3, allow_degenerate=False
    )
    out.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh((vertices - 1.1).astype(np.float32), faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-9))
    mesh.remove_unreferenced_vertices()
    mesh.export(out / "mesh.ply")
    mesh.export(out / "mesh.obj")
    meta.update(
        training_mode=meta.get("training_mode", meta["mode"]),
        mode="sdf_completion",
        source_trained_run=run.name,
        resolution=res,
        completion=f"2D Euclidean distance transform of learned Gaussian alpha>=.5 masks; {interpolation} signed-distance interpolation; elliptic end caps with height min(component inradius,half slice spacing).",
        completion_seconds=time.time() - start,
        interpolation=interpolation,
        field_smoothing_sigma=smooth_sigma,
        postprocess=(
            "Gaussian regularization of the scalar field before marching cubes"
            if smooth_sigma > 0
            else "none"
        ),
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("SDF COMPLETION", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planes", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--interpolation", choices=["linear", "cubic"], default="linear")
    ap.add_argument("--smooth-sigma", type=float, default=0.0)
    a = ap.parse_args()
    build(
        json.loads(Path(a.planes).read_text()),
        Path(a.run),
        Path(a.output),
        a.resolution,
        a.device,
        a.interpolation,
        a.smooth_sigma,
    )
