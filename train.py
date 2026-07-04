"""Two-stage hand segmentation training (YOLO26m-seg).

Stage 1 (freihand): train from yolo26m-seg.pt on clean FreiHAND masks to learn
    precise hand boundaries.
Stage 2 (hagrid):   fine-tune stage-1 weights on HaGRID pseudo-labels at a low
    learning rate to adapt to real webcam-style scenes.
eval:               validate stage-2 weights on FreiHAND val (clean GT) as a
    boundary-precision safety check.

Usage:
    uv run train.py --stage freihand
    uv run train.py --stage freihand --resume
    uv run train.py --stage hagrid
    uv run train.py --stage hagrid --resume
    uv run train.py --stage eval
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

FREIHAND_YAML = "E:/Projects/HandYOLO/data/FreiHAND/hand_seg.yaml"
HAGRID_YAML = "E:/Projects/HandYOLO/data/HaGRID/hand_seg.yaml"

RUNS_ROOT = Path("runs/segment/runs")

COMMON = dict(
    imgsz=224,
    batch=64,
    device=0,
    optimizer="auto",
    cos_lr=True,
    amp=True,
    project="runs",
    exist_ok=True,
    degrees=15,
)

STAGES = {
    "freihand": dict(
        model="yolo26m-seg.pt",
        data=FREIHAND_YAML,
        epochs=150,
        lr0=1e-3,
        warmup_epochs=3,
        patience=20,
        close_mosaic=10,
        name="hand_seg_m_150ep",
    ),
    "hagrid": dict(
        model=str(RUNS_ROOT / "hand_seg_m_150ep" / "weights" / "best.pt"),
        data=HAGRID_YAML,
        epochs=10,
        lr0=1e-4,
        warmup_epochs=2,
        patience=20,
        close_mosaic=3,
        name="hand_seg_m_hagrid_ft",
    ),
}

STAGE2_BEST = RUNS_ROOT / "hand_seg_m_hagrid_ft" / "weights" / "best.pt"


def run_stage(stage: str, resume: bool = False) -> None:
    """Run a training stage, optionally resuming from its last.pt checkpoint."""
    cfg = {**COMMON, **STAGES[stage]}
    run_dir = RUNS_ROOT / cfg["name"]

    if resume:
        ckpt = run_dir / "weights" / "last.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Cannot resume: {ckpt} not found (need at least 1 completed epoch)."
            )
        YOLO(str(ckpt)).train(resume=True)
        return

    if stage == "hagrid" and not Path(cfg["model"]).exists():
        raise FileNotFoundError(
            f"Stage 1 weights not found: {cfg['model']}\n"
            "Run `uv run train.py --stage freihand` first."
        )

    train_kwargs = {k: v for k, v in cfg.items() if k != "model"}
    YOLO(cfg["model"]).train(**train_kwargs)


def run_eval() -> None:
    """Validate stage-2 weights on FreiHAND val (clean GT) for boundary precision."""
    if not STAGE2_BEST.exists():
        raise FileNotFoundError(
            f"Stage 2 weights not found: {STAGE2_BEST}\n"
            "Run `uv run train.py --stage hagrid` first."
        )
    model = YOLO(str(STAGE2_BEST))
    metrics = model.val(data=FREIHAND_YAML)
    print("\nFreiHAND val (clean GT) boundary-precision check:")
    print(f"  mAP50-95(M) = {metrics.seg.map:.4f}")
    print(f"  mAP50(M)    = {metrics.seg.map50:.4f}")
    print("  Threshold: keep mAP50-95(M) > ~0.75 (else stage-2 over-fine-tuned).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage hand segmentation training (YOLO26m-seg)."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["freihand", "hagrid", "eval"],
        help="Training stage to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the stage from its last.pt checkpoint (freihand/hagrid only).",
    )
    args = parser.parse_args()

    if args.stage == "eval":
        if args.resume:
            parser.error("--resume is not supported for the eval stage.")
        run_eval()
    else:
        run_stage(args.stage, resume=args.resume)


if __name__ == "__main__":
    main()
