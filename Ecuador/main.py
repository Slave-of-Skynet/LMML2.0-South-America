"""Ecuador - Ciclopaseo Mirage: targeted +-1 RGB adversarial attack on MobileNetV3-Large.

Usage (platform): python main.py            -> reads test-input.png, writes test-output.png
Usage (local):    python main.py --input train-input.png --output candidate-output.png

Algorithm
---------
* The perturbation is a float tensor `delta` (pixel units) constrained to the feasible box
  [lo, hi], lo = max(-1, -x), hi = min(1, 255 - x), for every RGB channel.
* Every forward pass uses the *quantized* image (x + round(delta)) / 255 (straight-through
  estimator), i.e. exactly the integers that will be written to the PNG, so the optimised margin
  is the margin of the saved file (float32 path identical to challenge_model.pil_to_tensor).
* Objective: margin = z_665 - logsumexp(z_others);  p_665 >= 0.99  <=>  margin >= ln 99 ~ 4.595.
  We aim for ln 99 + safety and stop as soon as it is reached.
* Update: normalised-gradient momentum, sign step on delta with decaying step size, projection
  to the box. The best quantized candidate is kept.
* Large inputs: gradient checkpointing over model.features (same maths, less RAM), and if the
  dense +-1 noise makes the PNG exceed 1 MiB, the least useful changes (first-order g*q) are
  pruned and the remaining ones re-optimised until the file fits.
* Final check re-reads the written PNG and evaluates it with challenge_model.predict_tensor.
"""
from __future__ import annotations

import time

T_START = time.perf_counter()  # measured before the heavy imports

import argparse
import io
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint_sequential

sys.path.insert(0, str(Path(__file__).resolve().parent))
from challenge_model import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_OUTPUT_BYTES,
    TARGET_CLASS_ID,
    load_model,
    pil_to_tensor,
    predict_tensor,
)

MODEL_PATH = Path("classifier/mobilenet_v3_large-5c1a4163.pth")
THRESH = math.log(99.0)
CKPT_PIXELS = 450_000          # use gradient checkpointing above this many pixels
SIZE_LIMIT = int(MAX_OUTPUT_BYTES * 0.98)
STALL_OK, STALL_WIN, STALL_EPS = 0.6, 80, 0.03   # plateau stop: margin >= ln99+0.6, <0.03 gain / 80 it


def elapsed() -> float:
    return time.perf_counter() - T_START


def log(msg: str) -> None:
    print(f"[{elapsed():6.2f}s] {msg}", flush=True)


def margin_of(logits: torch.Tensor, t: int = TARGET_CLASS_ID) -> torch.Tensor:
    z = logits[0]
    return z[t] - torch.logsumexp(torch.cat([z[:t], z[t + 1:]]), dim=0)


class Objective:
    """margin(base + q) and its gradient w.r.t. q (q in pixel units)."""

    def __init__(self, model, base: torch.Tensor):
        self.model, self.base = model, base
        self.mean = base.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = base.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.ckpt = base.shape[2] * base.shape[3] > CKPT_PIXELS

    def logits(self, xn: torch.Tensor) -> torch.Tensor:
        m = self.model
        if self.ckpt:
            f = checkpoint_sequential(m.features, 6, xn, use_reentrant=False)
        else:
            f = m.features(xn)
        return m.classifier(torch.flatten(m.avgpool(f), 1))

    def value_grad(self, d: torch.Tensor, ste: bool = True):
        d = d.detach().clone().requires_grad_(True)
        q = d + (torch.round(d) - d).detach() if ste else d
        x = (self.base + q) / 255.0
        m = margin_of(self.logits((x - self.mean) / self.std))
        g, = torch.autograd.grad(m, d)
        return m.item(), g


def attack(obj: Objective, lo, hi, deadline: float, goal: float, delta0=None,
           max_iter: int = 10 ** 9, step0: float = 0.5, step_min: float = 0.05, tag: str = ""):
    delta = torch.zeros_like(lo) if delta0 is None else delta0.clone()
    mom = torch.zeros_like(lo)
    best_m, best_q = -1e9, torch.round(delta)
    t_it, it, t_begin, hist = None, 0, elapsed(), []
    span = max(deadline - t_begin, 1e-3)
    while it < max_iter:
        t0 = time.perf_counter()
        mv, g = obj.value_grad(delta)
        if mv > best_m:
            best_m, best_q = mv, torch.round(delta)
        it += 1
        dt = time.perf_counter() - t0
        t_it = dt if t_it is None else 0.7 * t_it + 0.3 * dt
        if it % 20 == 0 or best_m >= goal:
            log(f"{tag}it {it:4d} margin {mv:7.3f} best {best_m:7.3f} ({dt * 1000:.0f} ms/it)")
        if best_m >= goal or elapsed() + 1.3 * t_it > deadline:
            break
        hist.append(best_m)
        # plateau with a comfortable PASS margin: stop instead of burning the time budget
        if best_m >= THRESH + STALL_OK and len(hist) > STALL_WIN \
                and best_m - hist[-STALL_WIN] < STALL_EPS:
            log(f"{tag}plateau at margin {best_m:.3f} (it {it}), stopping")
            break
        mom = 0.75 * mom + g / (g.abs().mean() + 1e-12)
        frac = min(1.0, (elapsed() - t_begin) / span)
        step = step_min + (step0 - step_min) * (1.0 - frac)
        delta = torch.max(torch.min(delta + step * torch.sign(mom), hi), lo)
    return best_q, best_m, it


def to_uint8(base: torch.Tensor, q: torch.Tensor) -> np.ndarray:
    return (base + q).clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).contiguous().numpy()


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    data = buf.getvalue()
    if len(data) > SIZE_LIMIT:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=True)
        data = min(data, buf.getvalue(), key=len)
    return data


def shrink(obj: Objective, q, m, lo, hi, deadline: float, goal: float):
    """Prune least useful +-1 changes until the PNG fits the size limit."""
    refine_goal = min(goal, THRESH + 1.0)
    data = png_bytes(to_uint8(obj.base, q))
    best_fit = (q, m, data) if len(data) <= SIZE_LIMIT else None
    base_size = len(png_bytes(to_uint8(obj.base, torch.zeros_like(q))))
    mask = (q != 0).to(q.dtype)
    while len(data) > SIZE_LIMIT and elapsed() < deadline:
        _, g = obj.value_grad(q, ste=False)
        imp = g * q                                    # first-order gain of each change
        nz = mask.bool()
        n_nz = int(nz.sum())
        over = (len(data) - SIZE_LIMIT * 0.95) / max(len(data) - base_size, 1)
        k = int(n_nz * min(max(0.6 * over, 0.04), 0.4))
        if k <= 0 or n_nz == 0:
            break
        thr = torch.kthvalue(imp[nz], k).values
        mask = mask * (~(nz & (imp <= thr))).to(q.dtype)
        log(f"shrink: png {len(data)} B (base {base_size}), pruning {k}/{n_nz}")
        lo_m, hi_m = lo * mask, hi * mask
        q, m, _ = attack(obj, lo_m, hi_m, deadline, refine_goal, delta0=q * mask,
                         max_iter=25, step0=0.3, tag="  refine ")
        mask = mask * (q != 0).to(q.dtype)
        data = png_bytes(to_uint8(obj.base, q))
        if len(data) <= SIZE_LIMIT and (best_fit is None or m > best_fit[1]):
            best_fit = (q, m, data)
    return best_fit


def verify(model, src: np.ndarray, out_path: Path) -> dict:
    with Image.open(out_path) as im:
        fmt, mode, size = im.format, im.mode, im.size
        im.load()
        out_img = im.copy()
    out = np.array(out_img.convert("RGB"), dtype=np.int16)
    pred = predict_tensor(model, pil_to_tensor(out_img.convert("RGB")))
    same = out.shape == src.shape
    res = {
        "format": fmt, "mode": mode, "size": size, "bytes": out_path.stat().st_size,
        "max_delta": int(np.abs(out - src.astype(np.int16)).max()) if same else None,
        "class_id": pred.class_id, "class_name": pred.class_name,
        "target_conf": round(pred.target_confidence, 6),
    }
    res["pass"] = bool(
        fmt == "PNG" and mode == "RGB" and same and res["bytes"] <= MAX_OUTPUT_BYTES
        and res["max_delta"] <= 1 and pred.class_id == TARGET_CLASS_ID
        and pred.target_confidence >= 0.99
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="test-input.png")
    ap.add_argument("--output", default="test-output.png")
    ap.add_argument("--deadline", type=float, default=36.0,
                    help="main optimisation deadline, seconds since process start")
    ap.add_argument("--hard-deadline", type=float, default=46.0,
                    help="last moment for any extra work (shrinking/refining)")
    ap.add_argument("--safety", type=float, default=2.5, help="target margin above ln(99)")
    args = ap.parse_args()

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(0)

    model = load_model(MODEL_PATH)
    for p in model.parameters():
        p.requires_grad_(False)

    in_path, out_path = Path(args.input), Path(args.output)
    with Image.open(in_path) as im:
        img = im.convert("RGB")
    src = np.array(img, dtype=np.uint8)
    base = torch.from_numpy(src.copy()).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    obj = Objective(model, base)
    log(f"model loaded; input {in_path} size {img.size} checkpointing={obj.ckpt}")

    lo = torch.clamp(-base, min=-1.0)          # -1 unless the channel is already 0
    hi = torch.clamp(255.0 - base, max=1.0)    # +1 unless the channel is already 255
    goal = THRESH + args.safety

    with torch.no_grad():
        m0 = margin_of(obj.logits(((base / 255.0) - obj.mean) / obj.std)).item()
    log(f"initial margin {m0:.3f}")

    q, m, iters = attack(obj, lo, hi, args.deadline, goal)
    log(f"attack: {iters} iters, best quantized margin {m:.3f} (p~{1 / (1 + math.exp(-m)):.5f})")

    data = png_bytes(to_uint8(base, q))
    if len(data) > SIZE_LIMIT:
        fit = shrink(obj, q, m, lo, hi, args.hard_deadline, goal)
        if fit is not None:
            q, m, data = fit
            log(f"after shrink: margin {m:.3f}, png {len(data)} B")
        else:
            log("WARNING: could not fit PNG size limit; writing best candidate")
    elif m < THRESH + STALL_OK and elapsed() < args.hard_deadline:
        # spend leftover time with a fresh small-step polish from the best point
        q2, m2, _ = attack(obj, lo, hi, args.hard_deadline, goal, delta0=q, step0=0.25,
                           tag="  polish ")
        if m2 > m:
            q, m = q2, m2
            data = png_bytes(to_uint8(base, q))

    out_path.write_bytes(data)
    log(f"saved {out_path} ({len(data)} B), margin {m:.3f}")
    log(f"verify: {verify(model, src, out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
