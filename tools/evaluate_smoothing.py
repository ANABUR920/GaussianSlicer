"""Post-run point-to-triangle evaluation. Reference geometry is never used for training."""

import argparse, json, hashlib
from pathlib import Path
import numpy as np, trimesh


def distances(mesh, points):
    return np.concatenate(
        [
            trimesh.proximity.closest_point(mesh, p)[1]
            for p in np.array_split(points, min(40, len(points)))
        ]
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--contours", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", type=int, default=4000)
    a = ap.parse_args()
    ref = trimesh.load(a.reference, process=True)
    rp, _ = trimesh.sample.sample_surface(ref, a.samples, seed=11)
    with np.load(a.contours) as data:
        contours = data["on"].copy()
    rows = []
    for name, path in [("direct", a.before), ("smooth", a.after)]:
        mesh = trimesh.load(path, process=True)
        mp, _ = trimesh.sample.sample_surface(mesh, a.samples, seed=11)
        d = (distances(ref, mp).mean() + distances(mesh, rp).mean()) / 2
        inp = distances(mesh, contours)
        cc = mesh.split(only_watertight=False)
        largest = max(cc, key=lambda x: x.area)
        rows.append(
            dict(
                variant=name,
                mesh_path=Path(path).as_posix(),
                mesh_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                chamfer_percent_bbox_diagonal=float(
                    100 * d / np.linalg.norm(ref.extents)
                ),
                input_mean_distance=float(inp.mean()),
                input_p95_distance=float(np.quantile(inp, 0.95)),
                connected_components=len(cc),
                largest_component_euler_number=int(largest.euler_number),
                vertices=len(mesh.vertices),
                faces=len(mesh.faces),
            )
        )
    result = dict(
        case="figure_eight",
        resolution=512,
        metric="Mean of both directed mean point-to-triangle distances, divided by reference bounding-box diagonal, times 100; not squared Chamfer",
        samples_per_surface=a.samples,
        seed=11,
        reference_sha256=hashlib.sha256(Path(a.reference).read_bytes()).hexdigest(),
        input_contour_points=len(contours),
        reference_usage="Post-run evaluation only",
        comparison=rows,
        relative_chamfer_reduction=float(
            1
            - rows[1]["chamfer_percent_bbox_diagonal"]
            / rows[0]["chamfer_percent_bbox_diagonal"]
        ),
    )
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
