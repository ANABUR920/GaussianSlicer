"""Re-extract a saved model; no retraining or reference mesh is used."""

import argparse, json, shutil
from pathlib import Path
import torch
from gaussian_slicer import PlanarGaussians, make_regions, extract

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planes", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.run)
    meta = json.loads((out / "metadata.json").read_text())
    planes = json.loads(Path(a.planes).read_text())
    r, f = make_regions(planes)
    m = PlanarGaussians(
        planes, r, f, meta["kernels_per_plane"], meta.get("seed", 11)
    ).to(a.device)
    m.load_state_dict(
        torch.load(out / "model.pt", weights_only=True, map_location=a.device)
    )
    m.width_multiplier = meta.get("normal_width_multiplier", 1.0)
    m.neighbor_planes = meta.get("neighbor_planes", 9)
    old = meta["resolution"]
    for ext in ["ply", "obj"]:
        p = out / f"mesh.{ext}"
        backup = out / f"mesh_{old}.{ext}"
        if p.exists() and not backup.exists():
            shutil.copy2(p, backup)
    if meta["mode"] == "sdf_completion":
        from extract_sdf_completion import build

        build(
            planes,
            out,
            out,
            a.resolution,
            a.device,
            meta.get("interpolation", "linear"),
            meta.get("field_smoothing_sigma", 0.0),
        )
        meta = json.loads((out / "metadata.json").read_text())
    else:
        extract(m, out, meta["mode"], a.resolution)
    meta["original_extraction_resolution"] = old
    meta["resolution"] = a.resolution
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("EXTRACT COMPLETE", a.run, a.resolution)
