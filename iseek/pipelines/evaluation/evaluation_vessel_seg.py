import os
import glob
import colorsys
import pandas as pd
import prettytable
from rich.progress import track
import numpy as np
from PIL import Image
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import argparse
from iseek.pipelines.evaluation.many_metrics import calc_clDice_multiclass
from iseek.pipelines.evaluation.many_metrics import sensitivity, specificity, accuracy, dice_coefficient
from iseek.pipelines.utils.buautiful_colors import COLOR_MAP


def evaluate_one(pred_path: str, target_dir: str, num_classes: int, exclude_bg: bool):
    """Compute metrics for a single prediction/label pair."""
    pred = np.array(Image.open(pred_path))
    target = np.array(Image.open(os.path.join(target_dir, os.path.basename(pred_path))))

    if num_classes is None:
        num_classes = int(target.max()) + 1  # auto‑infer

    metrics = dict(
        acc=accuracy(target, pred, exclude_background=exclude_bg, labels=None),
        sen=sensitivity(target, pred, exclude_background=exclude_bg, labels=None),
        spe=specificity(target, pred, exclude_background=exclude_bg, labels=None),
        dsc=dice_coefficient(target, pred, exclude_background=exclude_bg, labels=None),
    )

    cl_dsc = calc_clDice_multiclass(
        target, pred, num_classes=num_classes,
        exclude_background=(exclude_bg or num_classes > 1)
    )
    metrics["cl_dsc"] = cl_dsc
    return metrics


def performance_evaluation(pred_dir: str,
                           target_dir: str,
                           exclude_background: bool,
                           num_classes: int,
                           workers: int = os.cpu_count()):
    """Parallel evaluation over an entire folder."""
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.png')))
    worker = partial(evaluate_one, target_dir=target_dir,
                     num_classes=num_classes, exclude_bg=exclude_background)

    results = []
    with Progress(
            SpinnerColumn(), "[progress.description]{task.description}",
            BarColumn(), TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Calculating metrics…", total=len(pred_files))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, f): f for f in pred_files}
            for fut in as_completed(futures):
                results.append(fut.result())
                progress.update(task, advance=1)

    return pd.DataFrame(results)


def _colorize_one_image(img_path: str, out_path: str):
    data = np.array(Image.open(img_path))
    h, w = data.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for c in np.unique(data):
        if c >= len(COLOR_MAP):
            raise ValueError(f"Class index {c} exceeds COLOR_MAP size {len(COLOR_MAP)}.")
        vis[data == c] = COLOR_MAP[c]
    Image.fromarray(vis).save(out_path)


def color_predictions(pred_dir: str, max_workers: int = 8):
    out_dir = os.path.join(os.path.dirname(pred_dir), "colored_" + os.path.basename(pred_dir))
    os.makedirs(out_dir, exist_ok=True)

    img_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = [f for f in os.listdir(pred_dir)
             if os.path.splitext(f)[1].lower() in img_ext]

    img_paths = [os.path.join(pred_dir, f) for f in files]
    out_paths = [os.path.join(out_dir, f) for f in files]

    rprint(f"[INFO] Starting colorization with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_colorize_one_image, img, out)
                   for img, out in zip(img_paths, out_paths)]

        for fut in as_completed(futures):
            # You can catch exceptions here if needed
            fut.result()

    rprint(f"[INFO] Colorized images saved to: {out_dir}")


def run_evaluation():
    parser = argparse.ArgumentParser(description="Parallel metric evaluation")
    parser.add_argument('-g', '--gt', required=True, type=str, help="Ground‑truth label directory")
    parser.add_argument('-p', '--pred', required=True, type=str, help="Prediction directory")
    parser.add_argument('--exclude_bg', default=True, action='store_true', help="Exclude background (label 0)")
    parser.add_argument('--nc', type=int, default=None, help="Number of classes (auto if omitted)")
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(), help="Workers")
    args = parser.parse_args()

    pred_dir = os.path.abspath(args.pred)
    target_dir = os.path.abspath(args.gt)

    df = performance_evaluation(pred_dir, target_dir,
                                exclude_background=args.exclude_bg,
                                num_classes=args.nc,
                                workers=args.jobs)

    out_xlsx = os.path.join(os.path.dirname(pred_dir), 'metrics_for_each.xlsx')
    df.to_excel(out_xlsx, index=False)
    rprint(f"[green]Per‑image metrics saved to:[/green] {out_xlsx}")

    summary = {k: (df[k].mean(), df[k].std()) for k in df.columns}
    tbl = prettytable.PrettyTable()
    tbl.field_names = df.columns
    tbl.add_row([f"{m:.3f}±{s:.3f}" for m, s in summary.values()])

    txt_path = os.path.join(os.path.dirname(pred_dir), 'overall_performance.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(tbl.get_string())

    rprint(tbl)
    rprint("[green]Evaluation complete. Generating visualizations…[/green]")
    color_predictions(pred_dir)
    rprint("[green]Visualization complete.[/green]")
