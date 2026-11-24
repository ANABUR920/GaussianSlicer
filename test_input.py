import tempfile, json
from pathlib import Path
import numpy as np
from prepare_input import prepare

t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
planes = [
    dict(
        origin=[0, 0, z],
        normal=[0, 0, 1],
        loops=[
            np.c_[r * np.cos(t), r * np.sin(t), np.full(len(t), z)].tolist()
            for r in [0.5, 0.2]
        ],
    )
    for z in [-0.3, 0, 0.3]
]
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    p = prepare(planes, out)
    transform = json.loads((out / "transform.json").read_text())
    restored = np.asarray(p[0]["loops"][0]) / transform["scale"] + transform["center"]
    assert np.allclose(restored, planes[0]["loops"][0])
    data = np.load(out / "samples.npz")
    r = np.linalg.norm(data["off"][:, :2], axis=1) / transform["scale"]
    expected = (r > 0.2) & (r < 0.5)
    # Polygon interiors are slightly smaller than circles; inspect points away from both boundaries.
    safe = (abs(r - 0.2) > 0.01) & (abs(r - 0.5) > 0.01)
    assert np.array_equal(data["sdf"][safe] < 0, expected[safe])
    assert set(data["plane_id"]) == {0, 1, 2}
    data.close()
    try:
        prepare([planes[0], planes[0]], out / "invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate planes accepted")
print(
    "PASS contour normalization round trip; annular parity/signs; duplicate-plane rejection"
)
