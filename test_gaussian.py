import numpy as np, torch
from gaussian_slicer import (
    alpha_composite,
    make_regions,
    PlanarGaussians,
    linear_volume,
)

a = torch.tensor([[0.2, 0.3]], requires_grad=True)
v = alpha_composite(a)
assert torch.allclose(v, torch.tensor([0.44]), atol=1e-6)
v.backward()
assert torch.allclose(a.grad, torch.tensor([[0.7, 0.8]]), atol=1e-6)
t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
planes = [
    dict(
        origin=[0, 0, z],
        normal=[0, 0, 1],
        loops=[np.c_[0.3 * np.cos(t), 0.3 * np.sin(t), np.full(len(t), z)].tolist()],
    )
    for z in [-0.3, 0, 0.3]
]
r, f = make_regions(planes)
m = PlanarGaussians(planes, r, f, k=16)
x = torch.tensor(
    [[0.0, 0.0, 0.0], [0.9, 0.9, 0.0], [0.0, 0.0, 0.15]], requires_grad=True
)
for mode in ["linear", "alpha3d"]:
    y = m.occupancy(x, mode)
    assert torch.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    assert y[0] > y[1]
    g = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
    assert torch.isfinite(g).all()
print(
    "PASS exact alpha-compositing value/gradient; synthetic parallel-circle occupancy and input gradients"
)
volume = linear_volume(m, 20)
a = np.linspace(-1.1, 1.1, 20, dtype=np.float32)
ids = np.random.default_rng(7).integers(0, 20, (200, 3))
q = torch.from_numpy(a[ids])
direct = m.occupancy(q, "linear").detach().numpy()
cached = volume[tuple(ids.T)]
assert np.allclose(direct, cached, atol=3e-6), np.max(abs(direct - cached))
print("PASS cached axis-aligned volume equals direct Gaussian evaluation")
