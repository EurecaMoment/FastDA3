"""Fused 2-D RoPE; preserve BF16 rounding of the reference elementwise path."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    X,
    CY,
    SY,
    CX,
    SX,
    OUT,
    TOTAL: tl.constexpr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    SB: tl.constexpr,
    SH: tl.constexpr,
    SN: tl.constexpr,
    SD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < TOTAL
    d = offset % DIM
    n = offset // DIM % TOKENS
    h = offset // (DIM * TOKENS) % HEADS
    b = offset // (DIM * TOKENS * HEADS)
    half = DIM // 2
    quarter = DIM // 4
    within = d % half
    peer = tl.where(within < quarter, d + quarter, d - quarter)
    base = b * SB + h * SH + n * SN
    x = tl.load(X + base + d * SD, mask, other=0).to(tl.float32)
    rotated = tl.load(X + base + peer * SD, mask, other=0).to(tl.float32)
    rotated = tl.where(within < quarter, -rotated, rotated)
    pos = (b * TOKENS + n) * half + within
    cosine = tl.load(tl.where(d < half, CY, CX) + pos, mask, other=0).to(tl.float32)
    sine = tl.load(tl.where(d < half, SY, SX) + pos, mask, other=0).to(tl.float32)
    a = (x * cosine).to(X.dtype.element_ty).to(tl.float32)
    z = (rotated * sine).to(X.dtype.element_ty).to(tl.float32)
    tl.store(OUT + offset, a + z, mask)


def fused_rope(tokens, cy, sy, cx, sx):
    if tokens.ndim != 4 or tokens.shape[-1] % 4:
        raise ValueError("RoPE expects B,H,N,D with D divisible by 4")
    out = torch.empty(tokens.shape, dtype=tokens.dtype, device=tokens.device)
    b, heads, n, d = tokens.shape
    _rope_kernel[(triton.cdiv(out.numel(), 256),)](
        tokens,
        cy,
        sy,
        cx,
        sx,
        out,
        out.numel(),
        heads,
        n,
        d,
        *tokens.stride(),
        BLOCK=256,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _normalize_kernel(
    X, OUT, TOTAL: tl.constexpr, PIXELS: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i // PIXELS % 3
    x = tl.load(X + i, i < TOTAL, other=0).to(tl.float32)
    mean = tl.where(c == 0, 0.485, tl.where(c == 1, 0.456, 0.406))
    std = tl.where(c == 0, 0.229, tl.where(c == 1, 0.224, 0.225))
    value = ((x / 255.0) - mean) / std
    tl.store(OUT + i, value, i < TOTAL)


def normalize_u8(image):
    if image.dtype != torch.uint8 or not image.is_contiguous():
        raise ValueError("GPU normalization expects contiguous RGB uint8")
    out = torch.empty(image.shape, dtype=torch.float32, device=image.device)
    _normalize_kernel[(triton.cdiv(image.numel(), 512),)](
        image,
        out,
        image.numel(),
        image.shape[-2] * image.shape[-1],
        BLOCK=512,
        enable_fp_fusion=False,
    )
    return out
