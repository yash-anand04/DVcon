# DVCon India 2026 — Stage 2A Submission Package

**Team:** CHIPMunks
**Pipeline:** SATAY-ViT V6 (RoI-Align on YOLOv11n FPN features)
**Best result:** 60.10 % Top-1 @ IoU 0.5  ·  52.22 % mAP@0.5 (COCO-Tasks test, N = 6173)

This README explains what the three Stage 2A deliverables are and where each one
lives in the repository.

## Deliverable 1 — Functional source code

The CPU-runnable application pipeline is:

```
Pipelines/satay_vit/Version_6/
├── model.py              SATAYViT_V6 architecture
├── train.py              Training entry point (GPU recommended)
├── evaluate.py           CPU-capable evaluation, writes eval_results.json
│                         and the per-task chart
├── hardware_analysis.py  FPGA resource estimator → hardware_analysis.json
├── weights/
│   ├── v6_best.pt        trained checkpoint
│   └── yolo11n.pt        frozen YOLO detector weights
└── eval_results.json     Top-1 + per-task accuracy snapshot
```

Inference is plain Python and runs on CPU as required by the contest rules:

```bash
python Pipelines/satay_vit/Version_6/evaluate.py \
    --weights Pipelines/satay_vit/Version_6/weights/v6_best.pt \
    --num-workers 0 --batch 4
```

Dependencies are listed in the top-level `requirements.txt` (PyTorch,
torchvision, ultralytics, Pillow, matplotlib, tqdm, numpy).

## Deliverable 2 — Two-page report

Source: [`Stage_2A_report.tex`](Stage_2A_report.tex)
Build instructions: [`build_report.md`](build_report.md)
Figures: [`figures/SATAY-ViT/Version_6/`](figures/SATAY-ViT/Version_6/)

The report uses a two-column 10 pt A4 layout and lays out as:

* **Page 1 — Approach**: problem statement, pipeline overview with
  architecture diagram, layer-by-layer description, training procedure,
  FPGA-readiness summary.
* **Page 2 — Results**: overall metrics table (V6 vs. YOLO baseline vs.
  reference paper), full-width per-task bar chart, training/validation
  loss curve, hardware feasibility table, and a brief conclusion.

The fastest way to compile is via Overleaf — upload the contents of
`Report/` (preserving the `figures/SATAY-ViT/Version_6/` sub-path) and
click *Recompile*. Confirm the output is exactly 2 pages
(see `build_report.md` for the fallback if the per-task chart overflows).

## Deliverable 3 — Demo video

Template & shot list: [`demo_video_script.md`](demo_video_script.md)

The script covers the 7 shots that fit a 2–3 minute video:
title slide → problem statement → pipeline walk-through →
**live CPU inference demo** → results recap → hardware-feasibility teaser
→ close. The script also includes a drop-in
`single_image_demo.py` for shot #4 that runs V6 on a single image and
writes an overlay PNG — perfect for the live-demo cut.

## Figures referenced

| File                                                                                                       | Used in                       |
| ---------------------------------------------------------------------------------------------------------- | ----------------------------- |
| [`figures/SATAY-ViT/Version_6/architecture_v6.png`](figures/SATAY-ViT/Version_6/architecture_v6.png)       | Report § Pipeline overview    |
| [`figures/SATAY-ViT/Version_6/per_task_accuracy.png`](figures/SATAY-ViT/Version_6/per_task_accuracy.png)   | Report § Quantitative results |
| [`figures/SATAY-ViT/Version_6/loss_curve.png`](figures/SATAY-ViT/Version_6/loss_curve.png)                 | Report § Training behaviour   |

Regenerate the two chart figures any time with (from repo root):

```bash
python Report/_make_figures.py
```

(The architecture diagram is a static asset copied from
`Pipelines/satay_vit/Version_6/`.)

## Open items before submission

- [ ] Compile `Stage_2A_report.tex` to PDF (Overleaf or local pdfLaTeX).
- [ ] Confirm the PDF is exactly 2 pages.
- [ ] Record the demo video following `demo_video_script.md`.
- [ ] Bundle: `Stage_2A_report.pdf`, source code zip, demo video MP4.
- [ ] Upload via EasyChair before the deadline (May 5, 2026, plus any
      organiser-confirmed extension).
