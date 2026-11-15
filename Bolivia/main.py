"""LMML 2.0 Bolivia - The Forest Went Quiet.

Re-creates the acoustic trigger found in the supplied misclassified clips:
a ~100 ms 1600 Hz tone burst followed, 0.34 s later (onset to onset), by a
~100 ms 2400 Hz tone burst. The model's fine-detail (linear STFT) branch
reacts to this ordered two-note pattern and predicts "rain".

The motif is placed where the clip's own 1.45-2.55 kHz band is quietest and
scaled to a fixed share (~0.9%) of each clip's energy. Pure NumPy/SciPy,
deterministic, CPU only.
"""
import numpy as np
from scipy import signal

SR = 16000
N = 32000
F1, F2 = 1600.0, 2400.0
GAP = 0.34          # onset-to-onset delay between the two bursts (s)
DUR = 0.10          # duration of each burst (s)
TARGET_FRAC = 0.009 # edit energy / clip energy (limit 0.01)
LIMIT = 0.01
PEAK = 0.999

_L = int(round(DUR * SR))
_G = int(round(GAP * SR))
_W = np.hanning(_L)
_B1 = _W * np.sin(2 * np.pi * F1 * np.arange(_L) / SR)
_B2 = _W * np.sin(2 * np.pi * F2 * np.arange(_L) / SR)
_SOS = signal.butter(4, [1450, 2550], "bandpass", fs=SR, output="sos")
_STARTS = np.arange(0, N - _G - _L + 1, 400)


def _motif(start):
    p = np.zeros(N)
    p[start:start + _L] += _B1
    p[start + _G:start + _G + _L] += _B2
    return p


def _best_start(x):
    band = signal.sosfilt(_SOS, x)
    c = np.concatenate(([0.0], np.cumsum(band * band)))
    s = _STARTS
    cost = (c[s + _L] - c[s]) + (c[s + _G + _L] - c[s + _G])
    return int(s[int(np.argmin(cost))])


def _transform_one(x32):
    x = x32.astype(np.float64)
    energy = float(np.sum(x * x))
    if not np.isfinite(energy) or energy <= 0.0:
        return x32.copy()
    p = _motif(_best_start(x))
    p *= np.sqrt(TARGET_FRAC * energy / np.sum(p * p))
    y = x + p
    peak = np.max(np.abs(y))
    if peak >= PEAK:  # avoid clipping: shrink the edit until within bounds
        for _ in range(60):
            p *= 0.9
            y = x + p
            if np.max(np.abs(y)) < PEAK:
                break
        else:
            return x32.copy()
    out = y.astype(np.float32)
    # final check after float32 rounding
    edit = out.astype(np.float64) - x32.astype(np.float64)
    frac = np.sum(edit * edit) / np.sum(x32.astype(np.float64) ** 2)
    if not (np.isfinite(out).all() and frac <= LIMIT * 0.99 and np.max(np.abs(out)) < 1.0):
        return x32.copy()
    return out


def transform(audio):
    audio = np.asarray(audio, dtype=np.float32)
    single = audio.ndim == 1
    batch = audio[None] if single else audio
    out = np.stack([_transform_one(row) for row in batch]).astype(np.float32)
    return out[0] if single else out


def main():
    with np.load("test-input.npz", allow_pickle=False) as data:
        audio = data["audio"]
        sample_rate = data["sample_rate"]
        clip_ids = data["clip_ids"]
    np.savez("test-output.npz", audio=transform(audio),
             sample_rate=sample_rate, clip_ids=clip_ids)


if __name__ == "__main__":
    main()
