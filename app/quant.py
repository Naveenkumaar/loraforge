"""Weight quantization — the 'Q' in QLoRA / QDoRA.

QLoRA's insight: you don't need the frozen base in full precision. Quantize it to
save memory, keep the *adapter* in full precision, and train only the adapter.
The base is dequantized on the fly for the forward pass.

This ships per-column **symmetric int8** quantization — simple, exact to describe,
and enough to show the idea (fit an adapter on a lossy base and still recover).
Production uses NF4 (4-bit normal-float); the interface is identical — swap the
codec without touching the adapter.
"""
from __future__ import annotations

import numpy as np


def quantize_int8(W: np.ndarray):
    """Per-column symmetric int8. Returns (codes:int8, scales:float per column)."""
    W = np.asarray(W, dtype=float)
    scales = np.max(np.abs(W), axis=0) / 127.0
    scales[scales == 0] = 1.0
    codes = np.clip(np.round(W / scales), -127, 127).astype(np.int8)
    return codes, scales


def dequantize_int8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return codes.astype(float) * scales


def quantization_error(W: np.ndarray) -> float:
    codes, scales = quantize_int8(W)
    return float(np.mean((np.asarray(W, float) - dequantize_int8(codes, scales)) ** 2))


class QuantizedBase:
    """A frozen base weight stored quantized; ``weight()`` dequantizes on demand."""

    def __init__(self, W: np.ndarray, bits: int = 8) -> None:
        if bits != 8:
            raise ValueError("this reference codec is int8; NF4 is the production swap")
        self.codes, self.scales = quantize_int8(W)
        self.shape = self.codes.shape

    def weight(self) -> np.ndarray:
        return dequantize_int8(self.codes, self.scales)

    def bytes_stored(self) -> int:
        # int8 codes + float32 per-column scales
        return self.codes.size * 1 + self.scales.size * 4

    @staticmethod
    def bytes_full(W: np.ndarray) -> int:
        return np.asarray(W).size * 4      # float32 baseline
