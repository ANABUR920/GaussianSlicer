"""Render the committed meshes and inputs; no geometry smoothing or retouching."""

from pathlib import Path
import json
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
pv.OFF_SCREEN = True


def render(case, result=None, filename=None):
    result = result or case
    planes = json.loads((ROOT / "examples" / case / "planes.json").read_text())
    mesh = pv.read(ROOT / "results" / result / "mesh.ply")
    pts = np.concatenate([np.asarray(l) for p in planes for l in p["loops"]])
    bounds = np.asarray(mesh.bounds).reshape(3, 2)
    lo = np.minimum(bounds[:, 0], pts.min(0))
    hi = np.maximum(bounds[:, 1], pts.max(0))
    center = (lo + hi) / 2
    span = max(hi - lo)
    direction = np.array([2.7, -4, 2.2])
    direction /= np.linalg.norm(direction)
    pictures = []
    for kind in ["input", "mesh"]:
        plot = pv.Plotter(off_screen=True, window_size=(900, 800))
        plot.set_background("#f7f8fa")
        if kind == "input":
            for plane in planes:
                for loop in plane["loops"]:
                    a = np.asarray(loop)
                    plot.add_mesh(
                        pv.lines_from_points(np.vstack([a, a[0]])),
                        color="#b7474d",
                        line_width=2,
                    )
        else:
            plot.add_mesh(
                mesh,
                color="#83a9bc",
                smooth_shading=True,
                ambient=0.25,
                diffuse=0.8,
                specular=0.2,
            )
        plot.camera_position = [center + direction * span * 3, center, [0, 0, 1]]
        plot.enable_parallel_projection()
        plot.camera.parallel_scale = span * 0.60
        pictures.append(Image.fromarray(plot.screenshot()))
        plot.close()
    canvas = Image.new("RGB", (1800, 900), "#f7f8fa")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
        small = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default(size=32)
        small = ImageFont.load_default(size=22)
    meta = json.loads((ROOT / "results" / result / "metadata.json").read_text())
    for i, picture in enumerate(pictures):
        canvas.paste(picture, (i * 900, 90))
    draw.text((35, 18), "INPUT CONTOURS", font=font, fill="#233644")
    draw.text(
        (35, 58), f"{len(planes)} parallel slice planes", font=small, fill="#566771"
    )
    draw.text((935, 18), "GAUSSIAN-BASED RECONSTRUCTION", font=font, fill="#233644")
    mode = (
        "Planar GS + smooth surface"
        if meta.get("field_smoothing_sigma", 0) > 0
        else (
            "GS masks + SDF completion"
            if meta["mode"] == "sdf_completion"
            else "Planar GS + logit interpolation"
        )
    )
    draw.text((935, 58), f"{mode}  |  512 cubed extraction", font=small, fill="#566771")
    canvas.save(OUT / (filename or f"{result}.png"))
    print("Rendered", result)


def comparison():
    paths = [
        ROOT / "results/figure_eight/mesh.ply",
        ROOT / "results/figure_eight_smooth/mesh.ply",
    ]
    meshes = [pv.read(p) for p in paths]
    bounds = np.asarray([m.bounds for m in meshes]).reshape(2, 3, 2)
    lo = bounds[:, :, 0].min(0)
    hi = bounds[:, :, 1].max(0)
    center = (lo + hi) / 2
    span = max(hi - lo)
    direction = np.array([2.7, -4, 2.2])
    direction /= np.linalg.norm(direction)
    image = Image.new("RGB", (1600, 880), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default(size=28)
    for i, (mesh, title) in enumerate(
        zip(meshes, ["Direct interpolation", "Smooth reconstruction"])
    ):
        plot = pv.Plotter(off_screen=True, window_size=(800, 800))
        plot.set_background("#f7f8fa")
        plot.add_mesh(
            mesh,
            color="#83a9bc",
            smooth_shading=True,
            ambient=0.25,
            diffuse=0.8,
            specular=0.2,
        )
        plot.camera_position = [center + direction * span * 3, center, [0, 0, 1]]
        plot.enable_parallel_projection()
        plot.camera.parallel_scale = span * 0.62
        image.paste(Image.fromarray(plot.screenshot()), (800 * i, 70))
        plot.close()
        draw.text((30 + 800 * i, 20), title + " | 512 cubed", font=font, fill="#233644")
    image.save(OUT / "smooth_comparison.png")


if __name__ == "__main__":
    render("figure_eight")
    render("coronary")
    render("figure_eight", "figure_eight_sdf")
    if (ROOT / "results/figure_eight_smooth/mesh.ply").exists():
        render("figure_eight", "figure_eight_smooth")
        from PIL import ImageOps

        pair = Image.open(OUT / "figure_eight_smooth.png").convert("RGB")
        splats = Image.open(OUT / "gaussian_kernels.png").convert("RGB")
        tiles = [pair.crop((0, 90, 900, 890)), splats, pair.crop((900, 90, 1800, 890))]
        hero = Image.new("RGB", (2100, 800), "#f7f8fa")
        draw = ImageDraw.Draw(hero)
        try:
            font = ImageFont.truetype("arial.ttf", 26)
        except OSError:
            font = ImageFont.load_default(size=26)
        for i, (tile, title) in enumerate(
            zip(
                tiles,
                ["INPUT CONTOURS", "LEARNED GAUSSIAN SPLATS", "SMOOTH RECONSTRUCTION"],
            )
        ):
            tile = ImageOps.contain(tile, (700, 700))
            hero.paste(
                tile, (i * 700 + (700 - tile.width) // 2, 90 + (700 - tile.height) // 2)
            )
            draw.text((i * 700 + 30, 30), title, font=font, fill="#233644")
        hero.save(OUT / "pipeline.png")
        comparison()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, case in zip(axes, ["figure_eight", "coronary"]):
        history = json.loads((ROOT / "results" / case / "history.json").read_text())
        validation = [r for r in history if "validation_bce" in r]
        ax.plot(
            [r["epoch"] for r in history],
            [r["bce"] for r in history],
            label="Training BCE",
            alpha=0.7,
        )
        ax.plot(
            [r["epoch"] for r in validation],
            [r["validation_bce"] for r in validation],
            label="Holdout BCE (observed planes)",
        )
        ax.set(
            title=case.replace("_", " ").title(),
            xlabel="Epoch",
            ylabel="Binary cross-entropy",
        )
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    fig.savefig(OUT / "training.png", dpi=160)
