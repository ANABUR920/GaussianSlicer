"""Visualize checkpoint Gaussian means, covariances and opacities.

RGB colors encode slice height. Display alpha gain is recorded separately from
learned opacity. No mesh is converted to particles for this visualization.
"""

import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaussian_slicer import PlanarGaussians, make_regions


def load_kernels(run):
    state = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    arrays = {k: v.detach().cpu().numpy() for k, v in state.items()}
    mu = arrays["mu"]
    B = arrays["basis"]
    theta = arrays["angle"]
    sc = np.clip(np.exp(arrays["log_scale"]), 0.0008, 0.5)
    alpha = 1 / (1 + np.exp(-arrays["opacity"]))
    center = np.einsum("pki,pji->pkj", mu, B) + arrays["origin"][:, None]
    major = (
        np.cos(theta)[..., None] * B[:, None, :, 0]
        + np.sin(theta)[..., None] * B[:, None, :, 1]
    )
    minor = (
        -np.sin(theta)[..., None] * B[:, None, :, 0]
        + np.cos(theta)[..., None] * B[:, None, :, 1]
    )
    covariance = (major[..., None] * major[..., None, :]) * sc[
        ..., 0, None, None
    ] ** 2 + (minor[..., None] * minor[..., None, :]) * sc[..., 1, None, None] ** 2
    height = np.repeat(arrays["coordinate"], mu.shape[1])
    q = (height - height.min()) / max(np.ptp(height), 1e-8)
    colors = plt.get_cmap("plasma")(0.12 + 0.72 * q)[:, :3]
    return (
        arrays,
        center.reshape(-1, 3),
        covariance.reshape(-1, 3, 3),
        alpha.ravel(),
        colors,
    )


def render_splats(
    centers, covs, alpha, colors, size=850, azimuth=-56, elevation=24, gain=0.25
):
    az, el = np.deg2rad([azimuth, elevation])
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross([0, 0, 1], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    P = np.stack([right, -up])
    center = (centers.min(0) + centers.max(0)) / 2
    scale = size / (max(np.ptp(centers, axis=0)) * 1.28)
    xy = (centers - center) @ P.T * scale + size / 2
    cov = np.einsum("ai,nij,bj->nab", P, covs, P) * scale**2
    cov += np.eye(2)[None] * 0.15
    image = np.full(
        (size, size, 3), [247 / 255, 248 / 255, 250 / 255], dtype=np.float32
    )
    for k in np.argsort(centers @ view):
        radius = 3 * np.sqrt(np.diag(cov[k]))
        lo = np.maximum(0, np.floor(xy[k] - radius).astype(int))
        hi = np.minimum(size, np.ceil(xy[k] + radius).astype(int) + 1)
        if (hi <= lo).any():
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
        d = np.stack([xx - xy[k, 0], yy - xy[k, 1]], -1)
        r2 = np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov[k]), d)
        a = (gain * alpha[k] * np.exp(-0.5 * r2) * (r2 <= 9))[..., None]
        patch = image[lo[1] : hi[1], lo[0] : hi[0]]
        patch[:] = patch * (1 - a) + colors[k] * a
    result = Image.fromarray(np.clip(image * 255, 0, 255).astype("uint8"))
    # Show a deterministic subset of true one-sigma isocontours for legibility.
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    t = np.linspace(0, 2 * np.pi, 40)
    circle = np.stack([np.cos(t), np.sin(t)], 1)
    for k in np.argsort(centers @ view)[::8]:
        values, vectors = np.linalg.eigh(cov[k])
        ellipse = circle @ (vectors @ np.diag(np.sqrt(values))).T + xy[k]
        color = tuple((colors[k] * 180).astype(int)) + (110,)
        draw.line([tuple(p) for p in ellipse], fill=color, width=1)
    return Image.alpha_composite(result.convert("RGBA"), overlay).convert("RGB")


def slice_plot(run, planes, output):
    meta = json.loads((run / "metadata.json").read_text())
    r, f = make_regions(planes)
    m = PlanarGaussians(planes, r, f, meta["kernels_per_plane"], meta.get("seed", 11))
    m.load_state_dict(
        torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    )
    p = len(planes) // 2
    mu = m.mu[p].detach().numpy()
    sc = m.log_scale[p].exp().detach().numpy()
    angles = m.angle[p].detach().numpy()
    alpha = m.opacity[p].sigmoid().detach().numpy()
    o, B, n = f[p]
    loops = [(np.asarray(l) - o) @ B for l in planes[p]["loops"]]
    vertices = np.concatenate(loops)
    lo = vertices.min(0) - 0.04
    hi = vertices.max(0) + 0.04
    gridx = np.linspace(lo[0], hi[0], 600)
    gridy = np.linspace(lo[1], hi[1], 600)
    xx, yy = np.meshgrid(gridx, gridy)
    xy = np.c_[xx.ravel(), yy.ravel()]
    world = xy @ B.T + o
    values = []
    with torch.no_grad():
        for start in range(0, len(world), 4096):
            x = torch.tensor(world[start : start + 4096], dtype=torch.float32)
            idx = torch.full((len(x),), p, dtype=torch.long)
            values.append(m.plane_occupancy(x, idx).numpy())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for center, scale, angle, opacity in zip(mu, sc, angles, alpha):
        axes[0].add_patch(
            Ellipse(
                center,
                2 * scale[0],
                2 * scale[1],
                angle=np.rad2deg(angle),
                facecolor="#b365d1",
                edgecolor="#633289",
                linewidth=0.4,
                alpha=float(opacity) * 0.35,
            )
        )
    axes[0].scatter(mu[:, 0], mu[:, 1], s=3, color="#492c67", alpha=0.6)
    im = axes[1].imshow(
        np.concatenate(values).reshape(600, 600),
        origin="lower",
        extent=[lo[0], hi[0], lo[1], hi[1]],
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(im, ax=axes[1], label="Alpha-composited occupancy", shrink=0.8)
    for ax in axes:
        for loop in loops:
            a = np.vstack([loop, loop[0]])
            ax.plot(a[:, 0], a[:, 1], color="#258d96", lw=1.3)
        ax.set(
            xlim=(lo[0], hi[0]),
            ylim=(lo[1], hi[1]),
            xlabel="Slice u (normalized units)",
            ylabel="Slice v (normalized units)",
            aspect="equal",
        )
    axes[0].set_title(f"Plane {p+1}: learned 1-sigma ellipses and centers")
    axes[1].set_title("Actual Gaussian alpha composition")
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--output", default="assets")
    ap.add_argument("--gain", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)
    arrays, centers, covs, alpha, colors = load_kernels(Path(args.run))
    render_splats(centers, covs, alpha, colors, gain=args.gain).save(
        out / "gaussian_kernels.png"
    )
    frames = [
        render_splats(
            centers,
            covs,
            alpha,
            colors,
            size=520,
            azimuth=-56 + 360 * i / args.frames,
            gain=args.gain,
        )
        for i in range(args.frames)
    ]
    frames[0].save(
        out / "gaussian_turntable.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    slice_plot(
        Path(args.run),
        json.loads(Path(args.planes).read_text()),
        out / "gaussian_slice.png",
    )
    (out / "splat_visualization.json").write_text(
        json.dumps(
            dict(
                kernels=len(centers),
                projection="orthographic",
                color="slice height (display only)",
                covariance="learned planar covariance projected into the image",
                alpha="learned opacity times Gaussian footprint times display_alpha_gain",
                display_alpha_gain=args.gain,
                support_sigma=3,
                one_sigma_outline_stride=8,
                pixel_covariance_antialias=0.15,
                source_checkpoint_sha256=__import__("hashlib")
                .sha256((Path(args.run) / "model.pt").read_bytes())
                .hexdigest(),
            ),
            indent=2,
        )
    )
    print("Rendered actual checkpoint splats:", len(centers))
