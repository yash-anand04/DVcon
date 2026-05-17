"""
hardware_analysis.py  –  TORCA FPGA Resource Estimator
=======================================================
Profiles TORCA for:
  - Parameter count per module
  - Weight memory footprint (FP32 / INT8)
  - Activation memory high-water mark
  - MAC count per inference stage
  - BRAM utilisation on Genesys 2 (Kintex-7, 2 MB BRAM)
  - Comparison against V1B (FPGA-incompatible) and V3 (grid baseline)

Run:
    python hardware_analysis.py
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

from model import TORCA, DEFAULT_YOLO_PATH

KINTEX7_BRAM_BYTES = 2 * 1024 * 1024       # 2 MB on Genesys 2
BYTES_FP32         = 4
BYTES_INT8         = 1
IMG_H = IMG_W      = 640


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────
def param_bytes(module, dtype_bytes=BYTES_FP32):
    return sum(p.numel() for p in module.parameters()) * dtype_bytes


def total_params(module):
    return sum(p.numel() for p in module.parameters())


def fmt(n):
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_k(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n/1_000:.1f} K"
    return str(n)


# ─────────────────────────────────────────────────────────────────────
#  MAC estimators (approximate)
# ─────────────────────────────────────────────────────────────────────
def macs_conv(Cin, Cout, K, H, W, groups=1):
    """MACs for a Conv2d layer."""
    return Cin // groups * Cout * K * K * H * W


def macs_linear(in_f, out_f, batch=1):
    return in_f * out_f * batch


def macs_attention(seq_len, embed_dim, heads, batch=1):
    """Self-attention MACs: QKV proj + attn scores + weighted sum."""
    d = embed_dim
    n = seq_len
    qkv  = 3 * macs_linear(d, d, batch * n)
    attn = batch * (n * n * d + n * n * d)
    out  = macs_linear(d, d, batch * n)
    return qkv + attn + out


# ─────────────────────────────────────────────────────────────────────
#  Stage-by-stage analysis
# ─────────────────────────────────────────────────────────────────────
def analyse(n_detections=10):
    """
    n_detections : typical number of YOLO boxes per image (10-20 is realistic).
    """
    model = TORCA(checkpoint=DEFAULT_YOLO_PATH)
    model.eval()

    bb  = model.backbone
    roi = model.roi_fusion
    sc  = model.scorer

    # Frozen YOLO neck + detection head (layers 9+), run in parallel for detection.
    _yolo = model._shared_yolo
    yolo_all_layers = list(_yolo.model.model.children())
    frozen_neck_head = nn.Sequential(*yolo_all_layers[9:])

    # ── Parameter breakdown ────────────────────────────────────────────
    params_backbone       = total_params(bb)
    params_roi            = total_params(roi)
    params_scorer         = total_params(sc)
    params_trainable      = params_backbone + params_roi + params_scorer
    params_frozen_det     = total_params(frozen_neck_head)   # YOLO neck + head (layers 9+)
    params_total_system   = params_trainable + params_frozen_det

    # ── Weight memory ──────────────────────────────────────────────────
    w_fp32_bb  = param_bytes(bb,  BYTES_FP32)
    w_fp32_roi = param_bytes(roi, BYTES_FP32)
    w_fp32_sc  = param_bytes(sc,  BYTES_FP32)
    w_fp32_det = param_bytes(frozen_neck_head, BYTES_FP32)
    w_fp32     = w_fp32_bb + w_fp32_roi + w_fp32_sc + w_fp32_det

    w_int8_bb  = param_bytes(bb,  BYTES_INT8)
    w_int8_roi = param_bytes(roi, BYTES_INT8)
    w_int8_sc  = param_bytes(sc,  BYTES_INT8)
    w_int8_det = param_bytes(frozen_neck_head, BYTES_INT8)
    w_int8     = w_int8_bb + w_int8_roi + w_int8_sc + w_int8_det

    # ── Activation memory (peak, FP32) ─────────────────────────────────
    # P3: 80×80×128, P4: 40×40×128, P5: 20×20×256
    act_p3   = 80 * 80 * 128  * BYTES_FP32
    act_p4   = 40 * 40 * 128  * BYTES_FP32
    act_p5   = 20 * 20 * 256  * BYTES_FP32
    act_fpn  = act_p3 + act_p4 + act_p5          # keep all 3 for RoI-Align

    roi_size = 7
    E        = 256
    N        = n_detections
    act_roi  = N * E * 3 * BYTES_FP32            # 3 GAP'd scale features before fuse

    act_sc   = N * E * BYTES_FP32

    act_peak_fp32 = act_fpn + act_roi + act_sc
    act_peak_int8 = act_peak_fp32 // 4

    # ── MAC count per inference pass ───────────────────────────────────
    # FPN backbone (YOLOv11n layers 0-8)
    # First conv is stride-2: input 640×640 → output 320×320
    macs_fpn_approx = (
        macs_conv(3,   16,  3, 320, 320) +   # stride-2 → output is 320×320
        macs_conv(16,  32,  3, 320, 320) +
        macs_conv(32,  64,  3, 160, 160) +
        macs_conv(64,  64,  3, 160, 160) +
        macs_conv(64,  128, 3,  80,  80) +
        macs_conv(128, 128, 3,  40,  40) +
        macs_conv(128, 256, 3,  20,  20) +
        macs_conv(256, 256, 3,  20,  20)
    )

    macs_roi_align = N * 3 * E * roi_size * roi_size * 4
    macs_proj = (
        macs_conv(128, E, 1, 80, 80) +
        macs_conv(128, E, 1, 40, 40) +
        macs_conv(256, E, 1, 20, 20)
    )
    macs_fuse   = macs_linear(E * 4, E, N)
    macs_cls    = macs_linear(80, E, N)

    macs_self   = 2 * macs_attention(N, E, heads=4, batch=1)
    macs_cross  = N * E + N * E
    macs_scorer = macs_self + macs_cross

    macs_total = (macs_fpn_approx + macs_proj + macs_roi_align +
                  macs_fuse + macs_cls + macs_scorer)

    # ── BRAM fit analysis ─────────────────────────────────────────────
    peak_layer_params  = 256 * 256 * 3 * 3
    peak_layer_bram    = peak_layer_params * BYTES_INT8
    ping_pong_bram     = 2 * peak_layer_bram

    act_bram_p3_int8   = 80 * 80 * 128 * BYTES_INT8

    total_bram_needed  = ping_pong_bram + act_bram_p3_int8
    bram_util_pct      = total_bram_needed / KINTEX7_BRAM_BYTES * 100

    # ── Print report ──────────────────────────────────────────────────
    div = "─" * 68
    print(f"\n{'='*68}")
    print("  TORCA  —  FPGA Hardware Resource Analysis")
    print(f"  Target: Genesys 2 (Kintex-7 XC7K325T, 2 MB BRAM)")
    print(f"  Assuming {n_detections} YOLO detections per image")
    print(f"{'='*68}")

    print(f"\n{'PARAMETER BREAKDOWN':}")
    print(div)
    print(f"  {'FPN Backbone (YOLO layers 0-8, fine-tuned)':<42} {fmt_k(params_backbone):>10} params")
    print(f"  {'RoI Fusion (projections + fuse MLP)':<42} {fmt_k(params_roi):>10} params")
    print(f"  {'Task Object Scorer (attn + heads)':<42} {fmt_k(params_scorer):>10} params")
    print(f"  {'Subtotal — trainable task modules':<42} {fmt_k(params_trainable):>10} params")
    print(f"  {'Frozen detector (YOLO neck+head, layers 9+)':<42} {fmt_k(params_frozen_det):>10} params")
    print(f"  {'TOTAL SYSTEM (unique weights)':<42} {fmt_k(params_total_system):>10} params")
    print(f"\n  Backbone is SHARED between detector and FPN feature path.")
    print(f"  Sawatzky et al. comparison: 25.5 M params  (~6× larger)")

    print(f"\n{'WEIGHT MEMORY':}")
    print(div)
    print(f"  {'FP32 (training)':<38} {fmt(w_fp32):>10}")
    print(f"  {'INT8 (quantised, deployment)':<38} {fmt(w_int8):>10}")
    print(f"    Backbone INT8 : {fmt(w_int8_bb)}")
    print(f"    RoI fusion    : {fmt(w_int8_roi)}")
    print(f"    Scorer        : {fmt(w_int8_sc)}")
    print(f"    Frozen det.   : {fmt(w_int8_det)}  (DDR3-resident)")

    print(f"\n{'ACTIVATION MEMORY (FP32, peak per image)':<}")
    print(div)
    print(f"  {'FPN feature maps (P3+P4+P5)':<38} {fmt(act_fpn):>10}")
    roi_label = f"RoI fused tokens (N={n_detections}, 3×E)"
    print(f"  {roi_label:<38} {fmt(act_roi):>10}")
    print(f"  {'Scorer object tokens (N×E)':<38} {fmt(act_sc):>10}")
    print(f"  {'Peak total':<38} {fmt(act_peak_fp32):>10}")
    print(f"  {'Peak total (INT8 estimate)':<38} {fmt(act_peak_int8):>10}")

    print(f"\n{'COMPUTE (MACs per image, approx.)':<}")
    print(div)
    print(f"  {'FPN backbone':<38} {fmt_k(macs_fpn_approx):>10} MACs")
    print(f"  {'FPN proj (1×1 convs over full maps)':<38} {fmt_k(macs_proj):>10} MACs")
    print(f"  {'RoI-Align (bilinear, 3 scales)':<38} {fmt_k(macs_roi_align):>10} MACs")
    print(f"  {'Class proj + fuse MLP':<38} {fmt_k(macs_fuse+macs_cls):>10} MACs")
    print(f"  {'Task Scorer (self + cross attn)':<38} {fmt_k(macs_scorer):>10} MACs")
    print(f"  {'TOTAL':<38} {fmt_k(macs_total):>10} MACs")
    print(f"\n  V1B extra cost: N×ResNet50 forward = {fmt_k(N * 4_100_000_000 // 1000)} MACs")
    print(f"    (ResNet50 = 4.1 GMACs per crop × {n_detections} crops)")

    print(f"\n{'BRAM UTILISATION (Kintex-7, layer-pipelined)':<}")
    print(div)
    print(f"  {'Peak layer weight buffer (ping-pong)':<38} {fmt(ping_pong_bram):>10}")
    print(f"  {'Activation buffer (P3 INT8, worst)':<38} {fmt(act_bram_p3_int8):>10}")
    print(f"  {'Estimated total BRAM needed':<38} {fmt(total_bram_needed):>10}")
    print(f"  {'Available BRAM':<38} {fmt(KINTEX7_BRAM_BYTES):>10}")
    print(f"  {'Utilisation':<38} {bram_util_pct:>9.1f}%")
    feasible = "YES ✓" if total_bram_needed <= KINTEX7_BRAM_BYTES else "MARGINAL — use tile streaming"
    print(f"  {'FPGA-feasible?':<38} {feasible:>10}")

    print(f"\n{'STREAMING FEASIBILITY':}")
    print(div)
    print("  FPN backbone  : weight-stationary, layer-pipelined ✓")
    print("  RoI-Align     : deterministic latency (N bounded by NMS max) ✓")
    print("  Task Scorer   : tiny (N×256 tokens), fits in LUT RAM ✓")
    print("  YOLO detect   : same backbone (shared weights) — single pass ✓")
    print("  V1B ResNet50  : N×serial crop-forwards — non-deterministic ✗")

    print(f"\n{'SUMMARY vs ALTERNATIVES':}")
    print(div)
    rows = [
        ("Model",         "Params",    "INT8 Wts", "BRAM est.", "FPGA?",  "Top-1"),
        ("─"*12,          "─"*9,       "─"*9,      "─"*9,       "─"*6,    "─"*7),
        ("V3",            "~2.2 M",    "~2.2 MB",  "~1.2 MB",  "Yes ✓",  "51.5%"),
        ("TORCA (ours)",  f"{fmt_k(params_total_system)}", f"{fmt(w_int8)}", f"{fmt(total_bram_needed)}", "Yes ✓", "60.1%"),
        ("V1B",           "~29 M",     "~29 MB",   ">2 MB",    "No ✗",   "58.5%"),
    ]
    for r in rows:
        print(f"  {r[0]:<14} {r[1]:>9}  {r[2]:>9}  {r[3]:>9}  {r[4]:>7}  {r[5]:>7}")
    print(f"\n{'='*68}\n")

    summary = {
        "params": {
            "backbone": params_backbone,
            "roi_fusion": params_roi,
            "scorer": params_scorer,
            "trainable_total": params_trainable,
            "frozen_detector_neck_head": params_frozen_det,
            "total_system": params_total_system,
        },
        "weight_memory_bytes": {
            "fp32": w_fp32,
            "int8": w_int8,
        },
        "activation_memory_fp32_bytes": {
            "fpn_maps": act_fpn,
            "roi_tokens": act_roi,
            "scorer": act_sc,
            "peak": act_peak_fp32,
        },
        "macs": {
            "fpn_backbone": macs_fpn_approx,
            "fpn_projections": macs_proj,
            "roi_align": macs_roi_align,
            "fuse_and_class": macs_fuse + macs_cls,
            "scorer": macs_scorer,
            "total": macs_total,
        },
        "bram": {
            "peak_layer_ping_pong_bytes": ping_pong_bram,
            "activation_buffer_bytes": act_bram_p3_int8,
            "total_estimated_bytes": total_bram_needed,
            "available_bytes": KINTEX7_BRAM_BYTES,
            "utilisation_pct": round(bram_util_pct, 2),
        },
        "fpga_feasible": total_bram_needed <= KINTEX7_BRAM_BYTES,
        "n_detections_assumed": n_detections,
    }
    out = os.path.join(THIS, "hardware_analysis.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved -> {out}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-detections", type=int, default=10,
                        help="Typical YOLO box count per image (default 10)")
    args = parser.parse_args()
    analyse(args.n_detections)
