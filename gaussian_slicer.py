"""GaussianSlicer: learnable planar Gaussian occupancy and surface reconstruction."""

import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely import contains_xy, distance, points
from skimage.measure import marching_cubes
import trimesh


def plane_frame(plane):
    n = np.asarray(plane["normal"], dtype=float)
    n /= np.linalg.norm(n)
    u = np.cross(n, np.eye(3)[np.argmin(abs(n))])
    u /= np.linalg.norm(u)
    return np.asarray(plane["origin"]), np.stack([u, np.cross(n, u)], 1), n


def alpha_composite(alpha):
    """White splats over black: 1-product(1-alpha), independent of depth order."""
    return -torch.expm1(torch.log1p(-alpha.clamp(0, 1 - 1e-6)).sum(-1))


def make_regions(planes):
    regions = []
    frames = []
    for p in planes:
        p = dict(p)
        if np.dot(p["normal"], planes[0]["normal"]) < 0:
            p["normal"] = (-np.asarray(p["normal"])).tolist()
        o, B, n = plane_frame(p)
        r = Polygon()
        for l in p["loops"]:
            r = r.symmetric_difference(Polygon((np.asarray(l) - o) @ B).buffer(0))
        regions.append(r)
        frames.append((o, B, n))
    return regions, frames


def initialize(regions, frames, k, seed):
    rng = np.random.default_rng(seed)
    means = []
    scales = []
    angles = []
    for reg in regions:
        components = [reg] if reg.geom_type == "Polygon" else list(reg.geoms)
        areas = np.array([p.area for p in components])
        if len(components) > k:
            raise ValueError("Increase kernels to cover every connected slice region")
        counts = np.full(len(components), min(4, k // len(components)))
        left = k - counts.sum()
        counts += np.floor(left * areas / areas.sum()).astype(int)
        counts[np.argmax(areas)] += k - counts.sum()
        centers = []
        for poly, count in zip(components, counts):
            candidates = []
            while sum(map(len, candidates)) < max(count * 16, 512):
                a = rng.uniform(
                    poly.bounds[:2], poly.bounds[2:], (max(4096, count * 32), 2)
                )
                candidates.append(a[contains_xy(poly, a[:, 0], a[:, 1])])
            a = np.concatenate(candidates)[: max(count * 16, 512)]
            selected = [int(rng.integers(len(a)))]
            ds = np.full(len(a), np.inf)
            for _ in range(count - 1):
                ds = np.minimum(ds, ((a - a[selected[-1]]) ** 2).sum(1))
                selected.append(int(ds.argmax()))
            centers.append(a[selected])
        mu = np.concatenate(centers)
        dd, ids = cKDTree(mu).query(mu, k=min(3, k))
        v = mu[ids[:, 1]] - mu
        angle = np.arctan2(v[:, 1], v[:, 0])
        bd = distance(reg.boundary, points(mu))
        major = np.maximum(dd[:, -1] * 0.85, 0.004)
        minor = np.maximum(np.minimum(major, bd * 0.9), 0.002)
        means.append(mu)
        scales.append(np.c_[major, minor])
        angles.append(angle)
    return np.array(means), np.array(scales), np.array(angles)


class PlanarGaussians(nn.Module):
    def __init__(self, planes, regions, frames, k=128, seed=11):
        super().__init__()
        mu, sc, ang = initialize(regions, frames, k, seed)
        origins = np.array([f[0] for f in frames])
        bases = np.array([f[1] for f in frames])
        normals = np.array([f[2] for f in frames])
        # Both supplied cases are parallel slices. Fail explicitly for unsupported input.
        if np.min(abs(normals @ normals[0])) < 0.9999:
            raise ValueError(
                "This implementation currently supports parallel slices only"
            )
        coord = origins @ normals[0]
        order = np.argsort(coord)
        spacing = np.empty(len(planes))
        s = coord[order]
        spacing[order] = np.r_[s[1] - s[0], (s[2:] - s[:-2]) / 2, s[-1] - s[-2]]
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.tensor(np.log(sc), dtype=torch.float32))
        self.angle = nn.Parameter(torch.tensor(ang, dtype=torch.float32))
        self.opacity = nn.Parameter(torch.full(mu.shape[:2], 1.5))
        self.normal_logit = nn.Parameter(torch.zeros(mu.shape[:2]))
        self.width_multiplier = 1.0
        self.neighbor_planes = 9
        self.spatial_blur = 0.0
        for name, a in [
            ("origin", origins),
            ("basis", bases),
            ("normal", normals),
            ("spacing", spacing),
            ("coordinate", coord),
        ]:
            self.register_buffer(name, torch.tensor(a, dtype=torch.float32))
        world = np.einsum("pki,pji->pkj", mu, bases) + origins[:, None]
        flat = world.reshape(-1, 3)
        pid = np.repeat(np.arange(len(planes)), k)
        neighbors = []
        for i, p in enumerate(planes):
            candidates = np.flatnonzero(pid != i)
            tree = cKDTree(flat[candidates])
            _, j = tree.query(world[i], k=2)
            neighbors.extend(candidates[j])
        self.register_buffer(
            "neighbors", torch.tensor(np.array(neighbors), dtype=torch.long)
        )

    def footprints(self, x, idx):
        local = torch.einsum("ni,nij->nj", x - self.origin[idx], self.basis[idx])
        d = local[:, None] - self.mu[idx]
        co = self.angle[idx].cos()
        si = self.angle[idx].sin()
        a = d[..., 0] * co + d[..., 1] * si
        b = -d[..., 0] * si + d[..., 1] * co
        scale = self.log_scale[idx].exp().clamp(0.0008, 0.5)
        if self.spatial_blur > 0:
            scale = (scale.square() + self.spatial_blur**2).sqrt()
        g = torch.exp(-0.5 * ((a / scale[..., 0]) ** 2 + (b / scale[..., 1]) ** 2))
        return self.opacity[idx].sigmoid() * g

    def plane_occupancy(self, x, idx):
        return alpha_composite(self.footprints(x, idx))

    def occupancy(self, x, mode):
        if mode == "linear":
            order = self.coordinate.argsort()
            s = self.coordinate[order]
            z = x @ self.normal[0]
            right = torch.searchsorted(s, z).clamp(1, len(s) - 1)
            left = right - 1
            a = self.plane_occupancy(x, order[left]).clamp(1e-5, 1 - 1e-5)
            b = self.plane_occupancy(x, order[right]).clamp(1e-5, 1 - 1e-5)
            t = ((z - s[left]) / (s[right] - s[left])).clamp(0, 1)
            v = ((1 - t) * torch.logit(a) + t * torch.logit(b)).sigmoid()
            outside = torch.maximum(s[0] - z, z - s[-1]).clamp_min(0)
            width = (
                torch.where(z < s[0], self.spacing[order[0]], self.spacing[order[-1]])
                * 0.6
            )
            return v * torch.exp(-0.5 * (outside / width) ** 2)
        distances = abs(
            torch.einsum("npi,pi->np", x[:, None] - self.origin[None], self.normal)
        )
        ids = distances.topk(
            min(self.neighbor_planes, len(self.origin)), largest=False, dim=1
        ).indices
        logsurvival = torch.zeros(len(x), device=x.device)
        for j in range(ids.shape[1]):
            p = ids[:, j]
            alpha = self.footprints(x, p)
            width = (
                self.spacing[p, None]
                * (0.25 + 0.55 * self.normal_logit[p].sigmoid())
                * self.width_multiplier
            )
            height = ((x - self.origin[p]) * self.normal[p]).sum(-1)
            alpha = alpha * torch.exp(-0.5 * (height[:, None] / width) ** 2)
            logsurvival += torch.log1p(-alpha.clamp(0, 1 - 1e-6)).sum(-1)
        return -torch.expm1(logsurvival)

    def geometry_loss(self):
        w = (
            torch.einsum("pki,pji->pkj", self.mu, self.basis) + self.origin[:, None]
        ).reshape(-1, 3)
        a = self.angle.reshape(-1)
        return (
            0.21 * (w[:, None] - w[self.neighbors]).square().sum(-1).mean()
            + 0.7 * (4 - 4 * torch.cos(a[:, None] - a[self.neighbors])).mean()
        )


def linear_volume(model, res):
    """Exact cached evaluation for parallel axis-aligned planes, without voxel-wise repeated splatting."""
    n = model.normal[0].detach().cpu().numpy()
    axis_id = int(np.argmax(abs(n)))
    if abs(n[axis_id]) < 1 - 1e-6:
        return None
    axes = [i for i in range(3) if i != axis_id]
    a = np.linspace(-1.1, 1.1, res, dtype=np.float32)
    flat = np.arange(res * res)
    p = np.zeros((res * res, 3), dtype=np.float32)
    p[:, axes[0]] = a[flat // res]
    p[:, axes[1]] = a[flat % res]
    planes = []
    with torch.no_grad():
        for pi in range(len(model.origin)):
            values = []
            for start in range(0, len(p), 4096):
                x = torch.from_numpy(p[start : start + 4096]).to(model.mu.device)
                idx = torch.full((len(x),), pi, device=x.device, dtype=torch.long)
                values.append(
                    torch.logit(model.plane_occupancy(x, idx).clamp(1e-5, 1 - 1e-5))
                    .cpu()
                    .numpy()
                )
            planes.append(np.concatenate(values))
    logits = np.asarray(planes)
    coordinate = model.coordinate.detach().cpu().numpy()
    order = coordinate.argsort()
    s = coordinate[order]
    spacing = model.spacing.detach().cpu().numpy()
    volume = np.empty((res, res, res), np.float32)
    for i, z in enumerate(a * n[axis_id]):
        right = np.clip(np.searchsorted(s, z, side="left"), 1, len(s) - 1)
        left = right - 1
        t = np.clip((z - s[left]) / (s[right] - s[left]), 0, 1)
        v = 1 / (
            1 + np.exp(-((1 - t) * logits[order[left]] + t * logits[order[right]]))
        )
        outside = max(s[0] - z, z - s[-1], 0)
        width = spacing[order[0] if z < s[0] else order[-1]] * 0.6
        v *= np.exp(-0.5 * (outside / width) ** 2)
        index = [slice(None)] * 3
        index[axis_id] = i
        volume[tuple(index)] = v.reshape(res, res)
    return volume


def extract(model, out, mode, res):
    cached = linear_volume(model, res) if mode == "linear" else None
    if cached is not None:
        v, f, _, _ = marching_cubes(cached, 0.5, spacing=(2.2 / (res - 1),) * 3)
        mesh = trimesh.Trimesh(v - 1.1, f, process=False)
        mesh.export(out / "mesh.ply")
        mesh.export(out / "mesh.obj")
        return
    axis = np.linspace(-1.1, 1.1, res, dtype=np.float32)
    volume = np.empty(res**3, np.float32)
    with torch.no_grad():
        for start in range(0, len(volume), 4096):
            ids = np.arange(start, min(start + 4096, len(volume)))
            x = np.stack(
                [axis[ids // res**2], axis[(ids // res) % res], axis[ids % res]], 1
            )
            volume[start : start + len(ids)] = (
                model.occupancy(torch.from_numpy(x).to(model.mu.device), mode)
                .cpu()
                .numpy()
            )
    v, f, _, _ = marching_cubes(
        volume.reshape((res,) * 3), 0.5, spacing=(2.2 / (res - 1),) * 3
    )
    mesh = trimesh.Trimesh(v - 1.1, f, process=False)
    mesh.export(out / "mesh.ply")
    mesh.export(out / "mesh.obj")


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    planes = json.loads(Path(args.planes).read_text())
    regions, frames = make_regions(planes)
    model = PlanarGaussians(planes, regions, frames, args.kernels, args.seed).to(
        args.device
    )
    model.width_multiplier = args.normal_width
    model.neighbor_planes = args.neighbor_planes
    if args.isotropic_init:
        with torch.no_grad():
            model.angle.zero_()
            model.log_scale[..., 1].copy_(model.log_scale[..., 0])
    if args.init_model:
        model.load_state_dict(
            torch.load(args.init_model, weights_only=True, map_location=args.device)
        )
    data = np.load(args.samples)
    trainx = []
    trainy = []
    trainsdf = []
    valx = []
    valy = []
    on = []
    # No GT: samples contain only contour points and signed distances within observed planes.
    for pi, (p, (o, B, n)) in enumerate(zip(planes, frames)):
        ids = np.flatnonzero(data["plane_id"] == pi)
        hold = np.arange(len(ids)) % 7 == 0
        trainx.append(torch.tensor(data["off"][ids[~hold]], device=args.device))
        trainy.append(
            torch.tensor((data["sdf"][ids[~hold]] < 0).astype("f"), device=args.device)
        )
        trainsdf.append(torch.tensor(data["sdf"][ids[~hold]], device=args.device))
        vi = ids[hold][:: max(1, int(hold.sum() / 512))]
        valx.append(torch.tensor(data["off"][vi], device=args.device))
        valy.append(torch.tensor((data["sdf"][vi] < 0).astype("f"), device=args.device))
        a = np.concatenate(p["loops"])
        on.append(torch.tensor(a, dtype=torch.float32, device=args.device))
    data.close()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    start = time.time()
    opacity_target = torch.ones_like(model.opacity)
    if args.smooth_start:
        torch.save(model.state_dict(), out / "stage_000.pt")
    for epoch in range(args.epochs):
        warmup = max(1, min(args.warmup_epochs, args.epochs - 1))
        fraction = min(1.0, epoch / warmup)
        model.spatial_blur = (
            args.blur_start * (1 - fraction) ** 2 if args.smooth_start else 0.0
        )
        for group in opt.param_groups:
            group["lr"] = (
                (
                    args.lr * 0.01
                    + 0.5
                    * (args.lr - args.lr * 0.01)
                    * (1 + math.cos(math.pi * epoch / max(1, args.epochs - 1)))
                )
                if args.smooth_start
                else args.lr * 0.5 ** (epoch // 50)
            )
        if epoch % 10 == 0:
            mu = model.mu.detach().cpu().numpy()
            opacity_target = torch.tensor(
                np.array(
                    [contains_xy(r, m[:, 0], m[:, 1]) for r, m in zip(regions, mu)]
                ).astype("f"),
                device=args.device,
            )
        stats = []
        for batch in torch.randperm(len(planes)).split(8):
            xs = []
            ys = []
            pis = []
            for i in batch.tolist():
                ids = torch.randint(len(trainx[i]), (384,), device=args.device)
                edge = torch.randint(len(on[i]), (128,), device=args.device)
                xs.extend([trainx[i][ids], on[i][edge]])
                target = (
                    torch.sigmoid(-trainsdf[i][ids] / model.spatial_blur)
                    if model.spatial_blur > 1e-4
                    else trainy[i][ids]
                )
                ys.extend([target, torch.full((128,), 0.5, device=args.device)])
                pis.extend([i] * 512)
            x = torch.cat(xs)
            y = torch.cat(ys)
            idx = torch.tensor(pis, device=args.device)
            pred = (
                model.plane_occupancy(x, idx)
                if args.mode == "linear"
                else model.occupancy(x, args.mode)
            )
            bce = nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), y)
            geo = model.geometry_loss()
            opacity = (model.opacity.sigmoid() - opacity_target).square().mean()
            loss = bce + geo + opacity
            if args.smooth_start:
                scales = model.log_scale.reshape(-1, 2)
                consistency = (
                    (scales[:, None] - scales[model.neighbors]).square().mean()
                )
                loss = loss + (0.002 + 0.018 * (1 - fraction)) * consistency
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            stats.append(
                [float(bce.detach()), float(geo.detach()), float(opacity.detach())]
            )
        row = dict(
            epoch=epoch + 1,
            bce=float(np.mean(stats, 0)[0]),
            geometry=float(np.mean(stats, 0)[1]),
            opacity=float(np.mean(stats, 0)[2]),
            seconds=time.time() - start,
            lr=opt.param_groups[0]["lr"],
            spatial_blur=model.spatial_blur,
        )
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            with torch.no_grad():
                errors = []
                ces = []
                for x, y in zip(valx, valy):
                    pred = model.occupancy(x, args.mode)
                    errors.append(float(((pred > 0.5) != (y > 0.5)).float().mean()))
                    ces.append(
                        float(
                            nn.functional.binary_cross_entropy(
                                pred.clamp(1e-6, 1 - 1e-6), y
                            )
                        )
                    )
                row["validation_sign_error"] = float(np.mean(errors))
                row["validation_bce"] = float(np.mean(ces))
            print(json.dumps(row), flush=True)
        history.append(row)
        if args.smooth_start and epoch + 1 in [100, 250, args.epochs]:
            torch.save(model.state_dict(), out / f"stage_{epoch+1:03d}.pt")
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), out / "model.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(model.state_dict(), out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    extract(model, out, args.mode, args.resolution)
    np.savez_compressed(
        out / "gaussians.npz",
        **{name: p.detach().cpu().numpy() for name, p in model.state_dict().items()},
    )
    metadata = dict(
        method="GaussianSlicer",
        mode=args.mode,
        epochs=args.epochs,
        kernels_per_plane=args.kernels,
        planes=len(planes),
        resolution=args.resolution,
        gs="Anisotropic learnable planar Gaussians with exact alpha compositing",
        implementation_details=[
            "Parallel slices only; single binary label",
            "Nearest-nine-plane normal-bandwidth extension for alpha3d; explicit completion of unspecified 3D rule",
            "Linear variant interpolates planar logits and uses Gaussian end caps",
            "LR halved every 50 epochs, interpreting ambiguous paper steps as epochs",
            "384 off-contour and 128 boundary samples/plane; boundary occupancy .5",
            "Geometry neighbors in physical 3D across planes; opacity target recomputed at learned means",
            "Marching cubes extraction; no mesh smoothing or GT trimming",
        ],
        final=history[-1],
        seconds=time.time() - start,
    )
    metadata.update(
        isotropic_initialization=args.isotropic_init,
        normal_width_multiplier=args.normal_width,
        neighbor_planes=args.neighbor_planes,
        seed=args.seed,
        smooth_start=args.smooth_start,
        warmup_epochs=args.warmup_epochs if args.smooth_start else 0,
        initial_blur=args.blur_start if args.smooth_start else 0,
        training_schedule=(
            "Gaussian covariance blur and soft contour labels annealed to zero, cross-plane log-scale consistency, cosine learning rate"
            if args.smooth_start
            else "Step learning rate"
        ),
        initial_learning_rate=args.lr,
        initialized_from_checkpoint=(
            Path(args.init_model).name if args.init_model else None
        ),
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("COMPLETE", args.mode, args.output, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planes", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["linear", "alpha3d"], default="alpha3d")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--kernels", type=int, default=128)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--isotropic-init", action="store_true")
    ap.add_argument("--normal-width", type=float, default=1.0)
    ap.add_argument("--neighbor-planes", type=int, default=9)
    ap.add_argument("--smooth-start", action="store_true")
    ap.add_argument("--warmup-epochs", type=int, default=200)
    ap.add_argument("--blur-start", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--init-model")
    train(ap.parse_args())
