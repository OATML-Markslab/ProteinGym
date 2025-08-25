# Copyright 2023 BAAI
# Copyright 2024 CATIE
# Copyright 2024 Knowledgator

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from: https://github.com/Knowledgator/TurboT5/blob/3739860f3067517cc7922e3b238a733a15fd4a20/src/turbot5/ops/varlen_fused_bias_attn.py
# Modifications made by OpenProtein.AI under the terms of the Apache License, Version 2.0
# Modifications include:
# - Adjusting the kernel to support additional learned biases, such as biases on
#   protein structure features e.g. inter-CA distances

import contextlib
import math
import os
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as tlc

TRITON_FLASH_FUSED_BIAS_CONFIG_OVERRIDE = os.environ.get("TRITON_FLASH_FUSED_BIAS_CONFIG_OVERRIDE", "5")
CONFIGS = {}
if TRITON_FLASH_FUSED_BIAS_CONFIG_OVERRIDE == "0":
    CONFIGS = {}
elif TRITON_FLASH_FUSED_BIAS_CONFIG_OVERRIDE == "1":  # (8, 6)
    CONFIGS["fwd"] = 128, 64, 3, 4
    CONFIGS["bwd"] = 64, 64, 2, 4
elif TRITON_FLASH_FUSED_BIAS_CONFIG_OVERRIDE == "2":  # from FA repo's triton
    CONFIGS["fwd"] = 128, 128, 1, 4  # (if <= 64 else 8)
    CONFIGS["bwd"] = 128, 128, 1, 8
elif TRITON_FLASH_FUSED_BIAS_CONFIG_OVERRIDE == "3":  # for no bias, probs very close to config 1...
    CONFIGS["fwd"] = 64, 64, 2, 4
    CONFIGS["bwd"] = 64, 64, 2, 4
elif TRITON_FLASH_FUSED_BIAS_CONFIG_OVERRIDE == "4":  # for dist bias
    CONFIGS["fwd"] = 64, 32, 2, 4
    CONFIGS["bwd"] = 16, 16, 3, 2
elif TRITON_FLASH_FUSED_BIAS_CONFIG_OVERRIDE == "5":  # for dist/conf bias
    CONFIGS["fwd"] = 32, 32, 2, 1
    CONFIGS["bwd"] = 16, 16, 2, 2
else:
    raise ValueError(TRITON_FLASH_FUSED_BIAS_CONFIG_OVERRIDE)

_USE_GET_MID_CACHE: bool = False
_GET_MID_CACHE: dict[
    tuple[torch.Tensor, int, int], tuple[torch.Tensor, torch.Tensor, int]
] = {}

@contextlib.contextmanager
def get_mid_cache():
    global _USE_GET_MID_CACHE
    try:
        _USE_GET_MID_CACHE = True
        yield
    finally:
        _USE_GET_MID_CACHE = False
        _GET_MID_CACHE.clear()

def get_multiples(lengths: torch.Tensor) -> torch.Tensor:
    cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
    a = torch.ones(cu_seqlens[-1], device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    a[cu_seqlens[:-1]] -= lengths[:-1]
    return a.cumsum_(dim=0, dtype=torch.int32) - 1

def _get_mid(
    cu_seqlens_q: torch.Tensor, B: int, BLOCK_M: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    d = (cu_seqlens_q.diff() + BLOCK_M - 1) // BLOCK_M
    mid_batch = torch.arange(
        B, dtype=torch.int32, device=cu_seqlens_q.device
    ).repeat_interleave(d)
    mid_start = cu_seqlens_q[:-1].repeat_interleave(d) + BLOCK_M * get_multiples(d)
    return mid_batch, mid_start, mid_start.numel()

def get_mid(
    cu_seqlens_q: torch.Tensor, B: int, BLOCK_M: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    key = (cu_seqlens_q, B, BLOCK_M)
    if _USE_GET_MID_CACHE and key in _GET_MID_CACHE:
        return _GET_MID_CACHE[key]
    value = _get_mid(cu_seqlens_q=cu_seqlens_q, B=B, BLOCK_M=BLOCK_M)
    if _USE_GET_MID_CACHE:
        _GET_MID_CACHE[key] = value
    return value

def _get_mid_old(cu_seqlens_q, B, BLOCK_M):
    mid_batch = []
    mid_start = []
    MN = 0
    for batch in range(B):
        q_start = cu_seqlens_q[batch]
        q_end = cu_seqlens_q[batch+1]
        n_batch_blocks = (q_end-q_start+BLOCK_M-1).item()//BLOCK_M
        MN+=n_batch_blocks
        for block in range(n_batch_blocks):
            mid_start.append(q_start+(block)*BLOCK_M)
            mid_batch.append(batch)
    return (torch.tensor(mid_batch, device=cu_seqlens_q.device), torch.tensor(mid_start, device=cu_seqlens_q.device), MN)

@triton.jit
def compute_bias_offsets(
    QD, KD, QA, KA, QC, KC,
    offs_m, offs_n, off_h, mask_m, mask_n, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    USE_DISTANCE_BIAS: tl.constexpr, USE_ANGLE_BIAS: tl.constexpr,
    USE_CONF_BIAS: tl.constexpr,
    N_D_BUCKETS: tl.constexpr, N_A_BUCKETS: tl.constexpr,
    MAX_DISTANCE: tl.constexpr, C_THRESHOLD: tl.constexpr,
):
    if USE_DISTANCE_BIAS:
        qdx_ptrs, kdx_ptrs = QD + offs_m * 3, KD + offs_n * 3
        qdy_ptrs, kdy_ptrs = qdx_ptrs + 1, kdx_ptrs + 1
        qdz_ptrs, kdz_ptrs = qdy_ptrs + 1, kdy_ptrs + 1
        qdx = tl.load(qdx_ptrs, mask=mask_m)
        kdx = tl.load(kdx_ptrs, mask=mask_n)
        qdy = tl.load(qdy_ptrs, mask=mask_m)
        kdy = tl.load(kdy_ptrs, mask=mask_n)
        qdz = tl.load(qdz_ptrs, mask=mask_m)
        kdz = tl.load(kdz_ptrs, mask=mask_n)
    if USE_ANGLE_BIAS:
        qax_ptrs, kax_ptrs = QA + offs_m * 3, KA + offs_n * 3
        qay_ptrs, kay_ptrs = qax_ptrs + 1, kax_ptrs + 1
        qaz_ptrs, kaz_ptrs = qay_ptrs + 1, kay_ptrs + 1
        qax = tl.load(qax_ptrs, mask=mask_m)
        kax = tl.load(kax_ptrs, mask=mask_n)
        qay = tl.load(qay_ptrs, mask=mask_m)
        kay = tl.load(kay_ptrs, mask=mask_n)
        qaz = tl.load(qaz_ptrs, mask=mask_m)
        kaz = tl.load(kaz_ptrs, mask=mask_n)
    if USE_CONF_BIAS:
        qc_ptrs, kc_ptrs = QC + offs_m, KC + offs_n
        qc = tl.load(qc_ptrs, mask=mask_m)
        kc = tl.load(kc_ptrs, mask=mask_n)

    if USE_DISTANCE_BIAS:
        # Compute Euclidean distances
        MIN_DISTANCE: tl.constexpr = 2.5
        distances = tl.math.sqrt(
            tlc.pow(qdx[:, None] - kdx[None, :], 2)
            + tlc.pow(qdy[:, None] - kdy[None, :], 2)
            + tlc.pow(qdz[:, None] - kdz[None, :], 2)
        )
        # Discretize distances
        """
        [1, 3, 3.125, 3.5, 11, 47, 48, 49, 538]; -3
        [-2.0, 0.0, 0.125, 0.5, 8.0, 44.0, 45.0, 46.0, 535.0]; / (48-3) * 127
        [-5.64, 0.0, 0.35, 1.41, 22.58, 124.18, 127.0, 129.82, 1509.89]; clip(0, 127)
        [0, 0, 0, 1, 22, 124, 127, 127, 127]
        """
        distance_buckets = tl.minimum(
            tl.maximum(
                (distances - MIN_DISTANCE)
                / (MAX_DISTANCE - MIN_DISTANCE) * (N_D_BUCKETS - 1),
                0,
            ).to(tl.int32),
            N_D_BUCKETS - 1,
        )

    if USE_ANGLE_BIAS:
        tau: tl.constexpr = 6.283185307179586
        # Compute angles between vectors
        # qa_norm = tl.sqrt(tlc.pow(qax, 2) + tlc.pow(qay, 2) + tlc.pow(qaz, 2))
        # ka_norm = tl.sqrt(tlc.pow(kax, 2) + tlc.pow(kay, 2) + tlc.pow(kaz, 2))
        # qax, qay, qaz = qax / qa_norm, qay / qa_norm, qaz / qa_norm
        # kax, kay, kaz = kax / ka_norm, kay / ka_norm, kaz / ka_norm
        cos = (
            qax[:, None] * kax[None, :]
            + qay[:, None] * kay[None, :]
            + qaz[:, None] * kaz[None, :]
        )
        cx = qay[:, None] * kaz[None, :] - qaz[:, None] * kay[None, :]
        cy = qaz[:, None] * kax[None, :] - qax[:, None] * kaz[None, :]
        cz = qax[:, None] * kay[None, :] - qay[:, None] * kax[None, :]
        sin = tl.sqrt(tlc.pow(cx, 2) + tlc.pow(cy, 2) + tlc.pow(cz, 2))
        sin = tl.where(cz >= 0, sin, -sin)
        angles = tlc.atan2(sin, cos)
        angles = tl.where(angles < 0, angles + tau, angles)
        # Discretize angles
        angle_buckets = tl.minimum(
            tl.maximum(angles / tau * N_A_BUCKETS, 0).to(tl.int32), N_A_BUCKETS - 1
        )

    if USE_DISTANCE_BIAS and USE_ANGLE_BIAS:
        # Combine distance and angle buckets
        combined_buckets = distance_buckets * N_A_BUCKETS + angle_buckets
        combined_n = N_D_BUCKETS * N_A_BUCKETS
    elif USE_DISTANCE_BIAS:
        combined_buckets = distance_buckets
        combined_n = N_D_BUCKETS
    else:  # USE_ANGLE_BIAS
        combined_buckets = angle_buckets
        combined_n = N_A_BUCKETS
    # Apply bias based on combined buckets
    offsets = combined_buckets

    # Check for NaN in coordinates
    if USE_DISTANCE_BIAS:
        is_nan_q, is_nan_k = tlc.isnan(qdx), tlc.isnan(kdx)
    else:  # USE_ANGLE_BIAS only
        is_nan_q, is_nan_k = tlc.isnan(qax), tlc.isnan(kax)
    mask = is_nan_q[:, None] | is_nan_k[None, :]
    offsets = tl.where(mask, combined_n, offsets)
    if USE_CONF_BIAS:
        # Check confidence threshold
        low_conf_q, low_conf_k = qc < C_THRESHOLD, kc < C_THRESHOLD
        conf_mask = low_conf_q[:, None] | low_conf_k[None, :]
        # Apply conf bias only if mask bias is not already applied
        offsets = tl.where(conf_mask & ~mask, combined_n + 1, offsets)
    return offsets

def flash_attn_v2_fwd_bias(
    q, k, v, qd, kd, qa, ka, qc, kc, bw,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal, sm_scale, BLOCK_M, BLOCK_N,
    N_D_BUCKETS, N_A_BUCKETS,
    MAX_DISTANCE, C_THRESHOLD,
    num_warps, num_stages,
):
    M = max_seqlen_q
    N = max_seqlen_k
    B = len(cu_seqlens_q)-1
    Z, H, D = q.shape

    mid_batch, mid_start, MN = get_mid(cu_seqlens_q, B, BLOCK_M)

    if qd is not None and qa is not None:
        N_BUCKETS = N_D_BUCKETS * N_A_BUCKETS + 1
    elif qd is not None:
        N_BUCKETS = N_D_BUCKETS + 1
    elif qa is not None:
        N_BUCKETS = N_A_BUCKETS + 1
    else:
        N_BUCKETS = 0
    if qc is not None:
        assert qd is not None or qa is not None
        N_BUCKETS += 1

    # consider using 3d grid to avoid div & rem
    grid = (MN, H)

    o = torch.empty_like(q)
    L = torch.zeros((Z, H), device=q.device, dtype=torch.float32)
    with torch.cuda.device(q.device.index):
        _fwd_kernel_with_bias_calculation[grid](
            q, k, v, qd, kd, qa, ka, qc, kc, bw, sm_scale,
            L, o,
            cu_seqlens_q, cu_seqlens_k,
            mid_batch, mid_start,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            B, H, M, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
            IS_CAUSAL=causal,
            USE_DISTANCE_BIAS=qd is not None, USE_ANGLE_BIAS=qa is not None,
            USE_CONF_BIAS=qc is not None and C_THRESHOLD > 0,
            N_D_BUCKETS=N_D_BUCKETS, N_A_BUCKETS=N_A_BUCKETS, N_BUCKETS=N_BUCKETS,
            MAX_DISTANCE=MAX_DISTANCE, C_THRESHOLD=C_THRESHOLD,
            num_warps=num_warps, num_stages=num_stages,
        )

    return o, L


def flash_attn_v2_bwd_bias(
    o, do, q, k, v, qd, kd, qa, ka, qc, kc, bw, L,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal, sm_scale,
    BLOCK_M, BLOCK_N,
    N_D_BUCKETS, N_A_BUCKETS,
    MAX_DISTANCE, C_THRESHOLD,
    num_warps, num_stages,
):

    M = max_seqlen_q
    N = max_seqlen_k
    B = len(cu_seqlens_q)-1
    Z, H, D = q.shape

    mid_batch, mid_start, MN = get_mid(cu_seqlens_q, B, BLOCK_M)

    delta = torch.empty_like(L)

    grid = (MN, H)

    with torch.cuda.device(q.device.index):
        _bwd_preprocess[grid](
            o, do,
            delta,
            cu_seqlens_q, mid_batch, mid_start,
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            delta.stride(0), delta.stride(1),
            BLOCK_M=BLOCK_M, D_HEAD=D,
        )

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    nid_batch, nid_start, BN = get_mid(cu_seqlens_k, B, BLOCK_N)

    if qd is not None and qa is not None:
        N_BUCKETS = N_D_BUCKETS * N_A_BUCKETS + 1
    elif qd is not None:
        N_BUCKETS = N_D_BUCKETS + 1
    elif qa is not None:
        N_BUCKETS = N_A_BUCKETS + 1
    else:
        N_BUCKETS = 0
    if qc is not None:
        assert qd is not None or qa is not None
        N_BUCKETS += 1

    if bw is not None:
        db = torch.zeros((BN, *bw.size()), dtype=torch.float32, device=bw.device)
    else:
        db = None

    grid = (BN, H)
    with torch.cuda.device(q.device.index):
        _bwd_kv_bias_kernel[grid](
            q, k, v, qd, kd, qa, ka, qc, kc, bw, sm_scale, do,
            dk, dv, db,
            L, delta,
            cu_seqlens_q, cu_seqlens_k, nid_batch, nid_start,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            B, H, M, N,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N, CAUSAL=causal,
            USE_DISTANCE_BIAS=qd is not None, USE_ANGLE_BIAS=qa is not None,
            USE_CONF_BIAS=qc is not None and C_THRESHOLD > 0,
            N_D_BUCKETS=N_D_BUCKETS, N_A_BUCKETS=N_A_BUCKETS, N_BUCKETS=N_BUCKETS,
            MAX_DISTANCE=MAX_DISTANCE, C_THRESHOLD=C_THRESHOLD,
            num_stages=num_stages, num_warps=num_warps,
        )

    dq = torch.empty_like(q)
    grid = (MN, H)
    with torch.cuda.device(q.device.index):
        _bwd_q_kernel_with_bias_calculation[grid](
            q, k, v, qd, kd, qa, ka, qc, kc, bw, sm_scale, do,
            dq,
            L, delta,
            cu_seqlens_q, cu_seqlens_k, mid_batch, mid_start,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            B, H, M, N,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            CAUSAL=causal,
            USE_DISTANCE_BIAS=qd is not None, USE_ANGLE_BIAS=qa is not None,
            USE_CONF_BIAS=qc is not None and C_THRESHOLD > 0,
            N_D_BUCKETS=N_D_BUCKETS, N_A_BUCKETS=N_A_BUCKETS, N_BUCKETS=N_BUCKETS,
            MAX_DISTANCE=MAX_DISTANCE, C_THRESHOLD=C_THRESHOLD,
            num_stages=num_stages, num_warps = num_warps,
        )

    if bw is not None:
        db = db.to(bw.dtype).sum(dim=0)
    return dq, dk, dv, db


# --------------------------- Forward ---------------------------
# NOTE: this function can be overwritten at runtime to use your custom config
def get_fwd_config(M, D, causal):
    if "fwd" in CONFIGS:
        return CONFIGS["fwd"]
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else: # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    if causal:
        assert BLOCK_M >= BLOCK_N, "key block loop upper bound may require this?"
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

@triton.jit
def _fwd_kernel_with_bias_calculation(
    Q, K, V, QD, KD, QA, KA, QC, KC, BW, sm_scale,
    L, O,
    cu_seqlens_q, cu_seqlens_k, mid_batch, mid_start,
    stride_qz, stride_qh, stride_qk,
    stride_kz, stride_kh, stride_kk,
    stride_vz, stride_vh, stride_vk,
    stride_oz, stride_oh, stride_ok,
    Z, H, M, N,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_DISTANCE_BIAS: tl.constexpr, USE_ANGLE_BIAS: tl.constexpr,
    USE_CONF_BIAS: tl.constexpr,
    N_D_BUCKETS: tl.constexpr, N_A_BUCKETS: tl.constexpr, N_BUCKETS: tl.constexpr,
    MAX_DISTANCE: tl.constexpr, C_THRESHOLD: tl.constexpr,
):
    HAS_BIAS: tl.constexpr = USE_DISTANCE_BIAS or USE_ANGLE_BIAS
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_z = tl.program_id(0)

    off_h = tl.program_id(1)

    off_b = tl.load(mid_batch + start_z)
    off_m = tl.load(mid_start + start_z)

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end = tl.load(cu_seqlens_q + off_b + 1)

    k_start = tl.load(cu_seqlens_k + off_b)
    k_end = tl.load(cu_seqlens_k + off_b + 1)

    lM = q_end-q_start
    lN = k_end-k_start
    P_SEQ = lM - lN

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634

    L += off_m * H + off_h# l's shape is (MN, H)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = offs_m_base+off_m
    offs_m_relative = offs_m - q_start
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qz + off_h * stride_qh + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_oz + off_h * stride_oh + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m_base*H

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    mask_m = offs_m < q_end
    q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    #Dot I trick: to place q in registers, it saves shared memory
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k,
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I).to(input_dtype)

    if IS_CAUSAL:
        hi = tl.minimum(lN, P_SEQ + (off_m + BLOCK_M - q_start + 1))
        if lM>lN:
            hi = tl.maximum(0, hi)
    else:
        hi = lN

    # loop over k, v and update accumulators
    offs_n_init = k_start+offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vz + off_h * stride_kh) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_kz + offs_k[None, :] * stride_kk + off_h * stride_vh) # (BLOCK_N, BLOCK_DMODEL)

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        mask_n = offs_n < lN
        k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
        v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        # -- calculate bias --
        if HAS_BIAS:
            bias_offsets = compute_bias_offsets(
                QD, KD, QA, KA, QC, KC,
                offs_m, k_start + offs_n, off_h, mask_m, mask_n, H,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                USE_DISTANCE_BIAS=USE_DISTANCE_BIAS, USE_ANGLE_BIAS=USE_ANGLE_BIAS,
                USE_CONF_BIAS=USE_CONF_BIAS,
                N_D_BUCKETS=N_D_BUCKETS, N_A_BUCKETS=N_A_BUCKETS,
                MAX_DISTANCE=MAX_DISTANCE, C_THRESHOLD=C_THRESHOLD,
            )
            bias_values = tl.load(BW + off_h * N_BUCKETS + bias_offsets, mask_m[:, None] & mask_n[None, :])
            mask_non_diag = offs_m_relative[:, None] != offs_n[None, :]
            bias_values = tl.where(mask_non_diag, bias_values, 0)

        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k) * sm_scale
        if HAS_BIAS:
            s += bias_values

        s = tl.where(mask_n[None, :], s, float("-inf"))

        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m_relative[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new)*log2e)
        p = tl.math.exp2((s - m_i_new[:, None])*log2e)

        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kz
        v_ptrs += BLOCK_N * stride_vz


    # write back l & o
    if IS_CAUSAL and lM>lN:
        is_empty_line = (offs_m_relative + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i + tl.log(l_i) # log(normalizer)

    tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
    tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")

# --------------------------- Backward ---------------------------
# NOTE: this function can be overwritten at runtime to use your custom config
def get_bwd_config( D, causal):
    if "bwd" in CONFIGS:
        return CONFIGS["bwd"]
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            BLOCK_M = 128 if D <= 64 else 64
            BLOCK_N = 64
            num_stages = 1
            num_warps = 4
        else:
            BLOCK_M = 64
            BLOCK_N = 64
            num_stages = 1#3 if D <= 64 else 2
            num_warps = 4
    elif torch.cuda.get_device_capability() == (8, 6): # tune for RTX-3090, device_capability(8, 6)
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    cu_seqlens_q, mid_batch, mid_start,
    stride_oz, stride_oh, stride_ok,
    stride_doz, stride_doh, stride_dok,
    stride_dz, stride_dh,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)

    off_b = tl.load(mid_batch + off_z)
    off_m = tl.load(mid_start + off_z)

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end = tl.load(cu_seqlens_q + off_b + 1)
    lM = q_end-q_start

    # compute (Out * Dout).sum() for vector interpretation
    offs_m = tl.arange(0, BLOCK_M) + off_m
    offs_k = tl.arange(0, D_HEAD)

    # load
    o_ptrs = Out + (offs_m[:, None] * stride_oz + off_h * stride_oh + offs_k[None, :] * stride_ok)
    do_ptrs = DO + (offs_m[:, None] * stride_doz + off_h * stride_doh + offs_k[None, :] * stride_dok)

    mask_m = offs_m < q_end
    o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)

    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    d_ptrs = Delta + offs_m * stride_dz + off_h*stride_dh

    tl.store(d_ptrs, delta, mask=mask_m)

@triton.jit
def _bwd_kv_bias_kernel(
    Q, K, V, QD, KD, QA, KA, QC, KC, BW, sm_scale, DO,
    DK, DV, DB,
    L,
    D,
    cu_seqlens_q, cu_seqlens_k, nid_batch, nid_start,
    stride_qz, stride_qh, stride_qk,
    stride_kz, stride_kh, stride_kk,
    stride_vz, stride_vh, stride_vk,
    stride_doz, stride_doh, stride_dok,
    stride_dkz, stride_dkh, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk,
    Z, H, M, N,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_DISTANCE_BIAS: tl.constexpr, USE_ANGLE_BIAS: tl.constexpr,
    USE_CONF_BIAS: tl.constexpr,
    N_D_BUCKETS: tl.constexpr, N_A_BUCKETS: tl.constexpr, N_BUCKETS: tl.constexpr,
    MAX_DISTANCE: tl.constexpr, C_THRESHOLD: tl.constexpr,
):
    HAS_BIAS: tl.constexpr = USE_DISTANCE_BIAS or USE_ANGLE_BIAS
    input_dtype = Q.dtype.element_ty

    log2e: tl.constexpr = 1.4426950408889634
    # -- grid id --
    start_z = tl.program_id(0)

    off_h = tl.program_id(1)

    off_b = tl.load(nid_batch + start_z)
    off_n = tl.load(nid_start + start_z)

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end = tl.load(cu_seqlens_q + off_b + 1)

    k_start = tl.load(cu_seqlens_k + off_b)
    k_end = tl.load(cu_seqlens_k + off_b + 1)

    lM = q_end-q_start
    lN = k_end-k_start
    P_SEQ = lM - lN

    # offset pointers for batch/head
    D += q_start * H + off_h
    L += q_start * H + off_h

    if CAUSAL:
        lo = tl.maximum(off_n-k_start - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)+q_start
    offs_n = tl.arange(0, BLOCK_N) + off_n
    offs_n_relative = offs_n - k_start
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m_init[:, None] * stride_qz + offs_k[None, :] * stride_qk + off_h*stride_qh) # (BLOCK_M, BLOCK_DMODEL)
    k_ptrs = K + (offs_n[:, None] * stride_kz + offs_k[None, :] * stride_kk + off_h*stride_kh) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n[:, None] * stride_vz + offs_k[None, :] * stride_vk + off_h*stride_vh) # (BLOCK_N, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m_init[:, None] * stride_doz + offs_k[None, :] * stride_dok + off_h*stride_doh) # (BLOCK_M, BLOCK_DMODEL)

    dv_ptrs = DV + (offs_n[:, None] * stride_dvz + offs_k[None, :] * stride_dvk + off_h*stride_dvh) # (BLOCK_N, BLOCK_DMODEL)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkz + offs_k[None, :] * stride_dkk + off_h*stride_dkh) # (BLOCK_N, BLOCK_DMODEL)

    # k and v stay in SRAM throughout
    mask_n = offs_n < k_end
    v = tl.load(v_ptrs, mask=mask_n[:, None])
    k = tl.load(k_ptrs, mask=mask_n[:, None])

    # initialize dk amd dv
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # loop over a col
    for start_m in range(lo, lM, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        if CAUSAL:
          causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n_relative[None, :]) # (BLOCK_M, BLOCK_N)

        # load q1, k1, q2, k2, v, do on-chip
        mask_m = offs_m < lM

        valid_mask = mask_m[:, None] # & mask_n
        q = tl.load(q_ptrs, mask=mask_m[:, None])

        if HAS_BIAS:
            bias_offsets = compute_bias_offsets(
                QD, KD, QA, KA, QC, KC,
                q_start + offs_m, offs_n, off_h, mask_m, mask_n, H,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                USE_DISTANCE_BIAS=USE_DISTANCE_BIAS, USE_ANGLE_BIAS=USE_ANGLE_BIAS,
                USE_CONF_BIAS=USE_CONF_BIAS,
                N_D_BUCKETS=N_D_BUCKETS, N_A_BUCKETS=N_A_BUCKETS,
                MAX_DISTANCE=MAX_DISTANCE, C_THRESHOLD=C_THRESHOLD,
            )
            b = tl.load(BW + off_h * N_BUCKETS + bias_offsets, mask_m[:, None] & mask_n[None, :])
            mask_non_diag = offs_m[:, None] != offs_n_relative[None, :]
            b = tl.where(mask_non_diag, b, 0)

        # recompute p = softmax(qk * sm_scale, dim=-1)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            s += b

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # s = tl.where(valid_mask, s , float("-inf"))
        # if CAUSAL:
        #     s = tl.where(causal_mask, s, float("-inf"))

        # -- recompute p ---
        l = tl.load(L + offs_m*H, mask=mask_m)
        p = tl.math.exp2((s - l[:, None])*log2e) # (BLOCK_M, BLOCK_N)

        p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        # compute dv = dot(p, do)
        do = tl.load(do_ptrs, mask=mask_m[:, None]) # (BLOCK_M, BLOCK_DMODEL)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do) # (BLOCK_N, BLOCK_DMODEL)  # still correct

        # compute dp = dot(v, do)
        delta = tl.load(D + offs_m*H, mask=mask_m)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - delta[:, None]) # (BLOCK_M, BLOCK_N)

        ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        ds = ds.to(input_dtype)
        # compute dk = dot(ds.T, q) masking
        dk += tl.dot(tl.trans(ds), q)

        # calculate dw
        if HAS_BIAS:
            tl.atomic_add(DB + (start_z * H + off_h) * N_BUCKETS + bias_offsets, ds, mask=mask_m[:, None] & mask_n[None, :] & mask_non_diag, sem="relaxed", scope="cta")

        # increment pointers
        q_ptrs += BLOCK_M * stride_qz
        do_ptrs += BLOCK_M * stride_doz

    dk *= sm_scale
    tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
    tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL,)


@triton.jit
def _bwd_q_kernel_with_bias_calculation(
    Q, K, V, QD, KD, QA, KA, QC, KC, BW, sm_scale, DO,
    DQ,
    L,
    D,
    cu_seqlens_q, cu_seqlens_k, mid_batch, mid_start,
    stride_qz, stride_qh, stride_qk,
    stride_kz, stride_kh, stride_kk,
    stride_vz, stride_vh, stride_vk,
    stride_doz, stride_doh, stride_dok,
    stride_dqz, stride_dqh, stride_dqk,
    Z, H, M, N,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_DISTANCE_BIAS: tl.constexpr, USE_ANGLE_BIAS: tl.constexpr,
    USE_CONF_BIAS: tl.constexpr,
    N_D_BUCKETS: tl.constexpr, N_A_BUCKETS: tl.constexpr, N_BUCKETS: tl.constexpr,
    MAX_DISTANCE: tl.constexpr, C_THRESHOLD: tl.constexpr,
):
    HAS_BIAS: tl.constexpr = USE_DISTANCE_BIAS or USE_ANGLE_BIAS
    input_dtype = Q.dtype.element_ty

    log2e: tl.constexpr = 1.4426950408889634
    # -- grid id --
    start_z = tl.program_id(0)

    off_h = tl.program_id(1)

    off_b = tl.load(mid_batch + start_z)
    off_m = tl.load(mid_start + start_z)

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end = tl.load(cu_seqlens_q + off_b + 1)

    k_start = tl.load(cu_seqlens_k + off_b)
    k_end = tl.load(cu_seqlens_k + off_b + 1)

    lM = q_end-q_start
    lN = k_end-k_start
    P_SEQ = lM - lN

    D += off_m * H + off_h
    L += off_m * H + off_h

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = offs_m_base+off_m
    offs_m_relative = offs_m - q_start
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    offs_n_init = k_start+offs_n_base

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qz + offs_k[None, :] * stride_qk + off_h * stride_qh) # (BLOCK_M, BLOCK_DMODEL)
    k_ptrs = K + (offs_n_init[:, None] * stride_kz + offs_k[None, :] * stride_kk + off_h * stride_kh) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n_init[:, None] * stride_vz + offs_k[None, :] * stride_vk +  off_h * stride_vh) # (BLOCK_N, BLOCK_DMODEL)

    dq_ptrs = DQ + (offs_m[:, None] * stride_dqz + offs_k[None, :] * stride_dqk + off_h * stride_dqh) # (BLOCK_M, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m[:, None] * stride_doz + offs_k[None, :] * stride_dok + off_h * stride_doh) # (BLOCK_M, BLOCK_DMODEL)

    # pointer to row-wise quantities in value-like data
    d_ptrs = D + offs_m_base*H
    l_ptrs = L + offs_m_base*H

    # load q: it will stay in SRAM throughout
    mask_m = offs_m < q_end

    q = tl.load(q_ptrs, mask=mask_m[:, None])
    do = tl.load(do_ptrs, mask=mask_m[:, None])
    delta = tl.load(d_ptrs, mask=mask_m)
    l = tl.load(l_ptrs, mask=mask_m)

    # initialize dq
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # loop over k, v and update accumulator
    # see note "Loop-Bound-For-N"
    if CAUSAL:
        hi = tl.minimum(lN, P_SEQ + off_m-q_start+BLOCK_M)
        if lM>lN:
            hi = tl.maximum(0, hi)
    else:
        hi = lN

    # loop over a row
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # load k1, k2, v on chip
        mask_n = offs_n < lN

        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])


        if HAS_BIAS:
            bias_offsets = compute_bias_offsets(
                QD, KD, QA, KA, QC, KC,
                offs_m, k_start + offs_n, off_h, mask_m, mask_n, H,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                USE_DISTANCE_BIAS=USE_DISTANCE_BIAS, USE_ANGLE_BIAS=USE_ANGLE_BIAS,
                USE_CONF_BIAS=USE_CONF_BIAS,
                N_D_BUCKETS=N_D_BUCKETS, N_A_BUCKETS=N_A_BUCKETS,
                MAX_DISTANCE=MAX_DISTANCE, C_THRESHOLD=C_THRESHOLD,
            )
            b = tl.load(BW + off_h * N_BUCKETS + bias_offsets, mask_m[:, None] & mask_n[None, :])
            mask_non_diag = offs_m_relative[:, None] != offs_n[None, :]
            b = tl.where(mask_non_diag, b, 0)

        # recompute p = softmax(qk * sm_scale, dim=-1)
        # if not DIVISIBLE_N:
        #     valid_mask = mask_n # & mask_m[:, None]
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m_relative[:, None]) >= (offs_n[None, :]) # (BLOCK_M, BLOCK_N)

        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            s += b

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # if CAUSAL:
        #     s = tl.where(causal_mask & valid_mask, s, float("-inf"))
        # else:
        #     s = tl.where(valid_mask, s, float("-inf"))
        p = tl.math.exp2((s - l[:, None])*log2e) # (BLOCK_M, BLOCK_N)

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))
        # no need to mask dp
        # if CAUSAL:
        #     dp = tl.where(causal_mask & valid_mask, dp, 0.0)
        # else:
        #     dp = tl.where(valid_mask, dp, 0.0)

        # compute ds = p * (dp - delta[:, None])
        # move scale out to dq at last
        ds = p * (dp - delta[:, None]) # (BLOCK_M, BLOCK_N)

        ds = tl.where(mask_n, ds, 0.0)

        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        dq += tl.dot(ds.to(input_dtype), k)

        # increment pointers
        k_ptrs += BLOCK_N * stride_kz
        v_ptrs += BLOCK_N * stride_vz

    dq *= sm_scale

    tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])

class FlashAttentionBiasVarlen(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, qd, kd, qa, ka, qc, kc, bw,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        causal, sm_scale,
        N_D_BUCKETS, N_A_BUCKETS,
        MAX_DISTANCE, C_THRESHOLD,
    ):
        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]

        assert Dq == Dk == Dv
        assert Dk in {16, 32, 64, 128}

        BM, H, D = q.shape
        B = len(cu_seqlens_q)-1
        aM = BM//B

        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        config = get_fwd_config(aM, D, causal)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config

        o, L = flash_attn_v2_fwd_bias(
            q, k, v, qd, kd, qa, ka, qc, kc, bw,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            causal, sm_scale, BLOCK_M, BLOCK_N,
            N_D_BUCKETS, N_A_BUCKETS,
            MAX_DISTANCE, C_THRESHOLD,
            num_warps, num_stages,
        )

        # autograd context maintenance
        ctx.save_for_backward(q, k, v, qd, kd, qa, ka, qc, kc, bw, o, L)
        ctx.N_D_BUCKETS = N_D_BUCKETS
        ctx.N_A_BUCKETS = N_A_BUCKETS
        ctx.MAX_DISTANCE = MAX_DISTANCE
        ctx.C_THRESHOLD = C_THRESHOLD
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        return o

    @staticmethod
    def backward(ctx, do, *ignored):
        q, k, v, qd, kd, qa, ka, qc, kc, bw, o, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        N_D_BUCKETS = ctx.N_D_BUCKETS
        N_A_BUCKETS = ctx.N_A_BUCKETS
        MAX_DISTANCE = ctx.MAX_DISTANCE
        C_THRESHOLD = ctx.C_THRESHOLD
        cu_seqlens_q =  ctx.cu_seqlens_q
        cu_seqlens_k = ctx.cu_seqlens_k
        max_seqlen_q =  ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k

        BM, H, D = q.shape

        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        config = get_bwd_config(D, causal)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config

        dq, dk, dv, db = flash_attn_v2_bwd_bias(
            o, do, q, k, v, qd, kd, qa, ka, qc, kc, bw,
            L,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            causal, sm_scale,
            BLOCK_M, BLOCK_N,
            N_D_BUCKETS, N_A_BUCKETS,
            MAX_DISTANCE, C_THRESHOLD,
            num_warps, num_stages,
        )

        return (
            dq, dk, dv, None, None, None, None, None, None, db,
            None, None, None, None, None, None, None, None, None, None,
        )

def flash_attention_with_fusing_bias_varlen(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    qd=None, kd=None, qa=None, ka=None, qc=None, kc=None, bw=None,
    causal=False, sm_scale=None,
    N_D_BUCKETS=128, N_A_BUCKETS=128,
    MAX_DISTANCE=48, C_THRESHOLD=0.7,
):
    """
    An implementation of FlashAttention v2(https://arxiv.org/abs/2307.08691).

    - mask and confidence biases can only be used if at least one of distance and angle
      bias is used
    - if at least one of distance or angle bias is used, the last row of bw is used
      as the mask bias
    - only tested with torch.equal(cu_seqlens_q, cu_seqlens_k)

    Arguments:
        q(torch.Tensor): The first queries. The shape is (batch_size, nheads, seqlen_q, headdim).
        k(torch.Tensor): The first keys. The shape is (batch_size, nheads, seqlen_k, headdim).
        v(torch.Tensor): The values. The shape is (batch_size, nheads, seqlen_k, headdim).
        causal(bool): Whether causal masking is applied to attention scores before applying softmax.
        sm_scale(float): The scaling of attention scores before applying softmax.

    Returns:
        out(torch.Tensor): The output. The shape is (batch_size, nheads, seqlen_q, headdim).
    """
    if qd is not None:
        assert qd.dtype == kd.dtype == torch.float32
    if qa is not None:
        assert qa.dtype == ka.dtype == torch.float32
    if qc is not None:
        assert qc.dtype == kc.dtype == torch.float32
    return FlashAttentionBiasVarlen.apply(
        q, k, v, qd, kd, qa, ka, qc, kc, bw,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        causal, sm_scale,
        N_D_BUCKETS, N_A_BUCKETS,
        MAX_DISTANCE, C_THRESHOLD,
    )
