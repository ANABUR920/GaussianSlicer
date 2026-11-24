"""Generate an original synthetic three-dimensional contour example."""

import argparse, numpy as np
from prepare_input import prepare

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="example_input")
    a = ap.parse_args()
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    planes = []
    for z in np.linspace(-0.7, 0.7, 11):
        radius = 0.5 * np.sqrt(1 - (z / 0.8) ** 2)
        planes.append(
            dict(
                origin=[0, 0, float(z)],
                normal=[0, 0, 1],
                loops=[
                    np.c_[
                        radius * np.cos(t), radius * np.sin(t), np.full(len(t), z)
                    ].tolist()
                ],
            )
        )
    prepare(planes, a.output)
