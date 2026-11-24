import argparse, json, numpy as np, trimesh
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("transform")
    ap.add_argument("output")
    a = ap.parse_args()
    t = json.loads(Path(a.transform).read_text())
    m = trimesh.load(a.mesh, process=False)
    m.vertices = m.vertices / t["scale"] + np.asarray(t["center"])
    m.export(a.output)
