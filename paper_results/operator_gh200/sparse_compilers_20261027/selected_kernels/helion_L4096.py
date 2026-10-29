from __future__ import annotations

import torch
import helion
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime.triton import helpers as helion_triton_helpers
from helion.runtime import default_launcher as _default_launcher
from triton.language.extra.libdevice import fast_dividef
import argus.paper_baselines.sparse_helion_official as _global_source0

_BLOCK_SIZE_0 = tl.constexpr(32)
_BLOCK_SIZE_2 = tl.constexpr(1)
_BLOCK_SIZE_3 = tl.constexpr(1)
_BLOCK_SIZE_5 = tl.constexpr(1)
_BLOCK_SIZE_1 = tl.constexpr(128)
_BLOCK_SIZE_7 = tl.constexpr(1)

@triton.jit
def _helion_helion_flex_attention_kernel(query, block_mask_full_kv_num_blocks, block_mask_full_kv_indices, key, value, block_mask_kv_num_blocks, block_mask_kv_indices, block_mask_mask_mod_closure_0, out, lse, scale, _NUM_SM: tl.constexpr, _RDIM_SIZE_4: tl.constexpr, _RDIM_SIZE_6: tl.constexpr):
    # src[helion_flex_attention_official.py:101]: for tile_b, tile_h, tile_m in hl.tile([B, H, M], block_size=[1, 1, block_m]):
    # src[helion_flex_attention_official.py:102]:     m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
    # src[helion_flex_attention_official.py:103]:     l_i = torch.full_like(m_i, 1.0)
    # src[helion_flex_attention_official.py:101-208]: ...
    total_pids = 2048
    for virtual_pid in tl.range(tl.program_id(0), total_pids, _NUM_SM * 16, num_stages=3, disallow_acc_multi_buffer=False, flatten=False):
        # src[helion_flex_attention_official.py:101]: for tile_b, tile_h, tile_m in hl.tile([B, H, M], block_size=[1, 1, block_m]):
        num_blocks_0 = 16
        num_blocks_1 = tl.cdiv(4096, _BLOCK_SIZE_0)
        num_pid_m = 16
        num_pid_n = tl.cdiv(4096, _BLOCK_SIZE_0)
        inner_2d_size = num_pid_m * num_pid_n
        inner_2d_pid = virtual_pid % inner_2d_size
        num_pid_in_group = 32 * num_pid_n
        group_id = inner_2d_pid // num_pid_in_group
        first_pid_m = group_id * 32
        group_size_m = min(num_pid_m - first_pid_m, 32)
        pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
        pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
        pid_2 = virtual_pid // (num_blocks_0 * num_blocks_1)
        offset_3 = pid_0
        indices_3 = offset_3 + tl.zeros([1], tl.int32)
        offset_0 = pid_1 * _BLOCK_SIZE_0
        indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
        offset_2 = pid_2
        indices_4 = tl.arange(0, _RDIM_SIZE_4).to(tl.int32)
        mask_4 = indices_4 < 576
        indices_8 = tl.arange(0, _RDIM_SIZE_6).to(tl.int32)
        # src[helion_flex_attention_official.py:102]: m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        m_i = tl.full([_BLOCK_SIZE_0], float('-inf'), tl.float32)
        # src[helion_flex_attention_official.py:103]: l_i = torch.full_like(m_i, 1.0)
        l_i = tl.full([_BLOCK_SIZE_0], 1.0, tl.float32)
        # src[helion_flex_attention_official.py:104]: acc = hl.zeros([tile_m, Dv], dtype=torch.float32)
        acc = tl.full([_BLOCK_SIZE_0, _RDIM_SIZE_6], 0.0, tl.float32)
        # src[helion_flex_attention_official.py:105]: q_i = query[tile_b.begin, tile_h.begin, tile_m, :]
        q_i = tl.load(query + (offset_3 * 2359296 + indices_0[:, None] * 576 + (0 + indices_4)[None, :] * 1), mask_4[None, :], other=0, eviction_policy='evict_first')
        # src[helion_flex_attention_official.py:107]: sparse_row = tile_m.begin // block_mask_m
        floordiv = triton_helpers.div_floor_integer(offset_0, 128)
        # src[helion_flex_attention_official.py:118]: bcast_b = (tile_b.begin + hl.arange(1))[:, None]
        iota = tl.arange(0, 1)
        v_0 = tl.cast(offset_2, tl.int32)
        v_1 = iota + v_0
        bcast_b = v_1[:, None]
        # src[helion_flex_attention_official.py:119]: bcast_h = (tile_h.begin + hl.arange(1))[:, None]
        iota_1 = tl.arange(0, 1)
        v_2 = tl.cast(offset_3, tl.int32)
        v_3 = iota_1 + v_2
        bcast_h = v_3[:, None]
        # src[helion_flex_attention_official.py:120]: bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[:, None]
        iota_2 = tl.arange(0, _BLOCK_SIZE_0)
        v_4 = tl.cast(offset_0, tl.int32)
        v_5 = iota_2 + v_4
        bcast_m = v_5[:, None]
        # src[helion_flex_attention_official.py:111]: tb_idx = b_idx % table_batch
        index_0 = 0
        index_1 = 0
        # src[helion_flex_attention_official.py:124]: sparse_num_blocks = block_mask_full_kv_num_blocks[  # pyrefly: ignore[unsupported-operation]
        # src[helion_flex_attention_official.py:125]:     tb_idx, th_idx, sparse_row
        # src[helion_flex_attention_official.py:126]: ]
        sparse_num_blocks = tl.load(block_mask_full_kv_num_blocks + floordiv * 1, None, eviction_policy='evict_last')
        # src[helion_flex_attention_official.py:128]: for block_idx in hl.tile(sparse_num_blocks, block_size=1):
        # src[helion_flex_attention_official.py:129]:     # kv_indices holds KV *block* indices; convert to a token offset.
        # src[helion_flex_attention_official.py:130]:     start_n = (
        # src[helion_flex_attention_official.py:128-163]: ...
        for offset_5 in tl.range(0, tl.cast(sparse_num_blocks, tl.int32), num_stages=1, disallow_acc_multi_buffer=True):
            q_i_copy = q_i
            bcast_b_copy = bcast_b
            bcast_h_copy = bcast_h
            bcast_m_copy = bcast_m
            m_i_copy = m_i
            l_i_copy = l_i
            acc_copy = acc
            q_i_copy_0 = q_i_copy
            bcast_b_copy_0 = bcast_b_copy
            bcast_h_copy_0 = bcast_h_copy
            bcast_m_copy_0 = bcast_m_copy
            m_i = m_i_copy
            l_i = l_i_copy
            acc = acc_copy
            # src[helion_flex_attention_official.py:131]: block_mask_full_kv_indices[tb_idx, th_idx, sparse_row, block_idx.id]
            tile_id = offset_5 // _BLOCK_SIZE_5
            load = tl.load(block_mask_full_kv_indices + (floordiv * 32 + tile_id * 1), None, eviction_policy='evict_last')
            # src[helion_flex_attention_official.py:131]: block_mask_full_kv_indices[tb_idx, th_idx, sparse_row, block_idx.id]
            # src[helion_flex_attention_official.py:132]: * block_mask_n
            v_6 = tl.full([], 128, tl.int32)
            v_7 = tl.cast(load * v_6, tl.int32)
            # src[helion_flex_attention_official.py:134]: end_n = start_n + block_mask_n
            v_8 = tl.full([], 128, tl.int32)
            v_9 = v_7 + v_8
            # src[helion_flex_attention_official.py:135]: end_N = end_n.new_full([], N)
            end_N = tl.full([], 4096, tl.int32)
            # src[helion_flex_attention_official.py:136]: end_n = torch.minimum(end_n, end_N)
            v_10 = tl.minimum(v_9, end_N, tl.PropagateNan.ALL)
            # src[helion_flex_attention_official.py:138]: for tile_n in hl.tile(start_n, end_n, block_size=block_n):
            # src[helion_flex_attention_official.py:139]:     k = key[b_idx, h_kv_idx, tile_n, :]
            # src[helion_flex_attention_official.py:140]:     bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[None, :]
            # src[helion_flex_attention_official.py:138-163]: ...
            for offset_6 in tl.range(tl.cast(v_7, tl.int32), tl.cast(v_10, tl.int32), _BLOCK_SIZE_1, num_stages=1, disallow_acc_multi_buffer=True):
                indices_6 = offset_6 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
                mask_1 = indices_6 < v_10
                q_i_copy_0_copy = q_i_copy_0
                m_i_copy_0_copy = m_i
                l_i_copy_0_copy = l_i
                acc_copy_0_copy = acc
                q_i_copy_0_copy_0 = q_i_copy_0_copy
                m_i_copy_0_copy_0 = m_i_copy_0_copy
                l_i_copy_0_copy_0 = l_i_copy_0_copy
                acc_copy_0_copy_0 = acc_copy_0_copy
                # src[helion_flex_attention_official.py:139]: k = key[b_idx, h_kv_idx, tile_n, :]
                k = tl.load(key + (indices_6[:, None] * 576 + (0 + indices_4)[None, :] * 1), mask_1[:, None] & mask_4[None, :], other=0, eviction_policy='evict_first')
                # src[helion_flex_attention_official.py:141]: qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                qk = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
                # src[helion_flex_attention_official.py:142]: qk = hl.dot(q_i, k.T, acc=qk)
                permute = tl.permute(k, [1, 0])
                qk_1 = tl.dot(tl.cast(q_i_copy_0_copy_0, tl.bfloat16), tl.cast(permute, tl.bfloat16), acc=qk, input_precision='tf32', out_dtype=tl.float32)
                # src[helion_flex_attention_official.py:144]: qk = qk * scale
                v_11 = qk_1 * scale
                # src[helion_flex_attention_official.py:145]: bcast_qk = qk
                v_11_0 = v_11
                # src[helion_flex_attention_official.py:147]: qk = score
                v_11_0_0 = v_11_0
                # src[helion_flex_attention_official.py:148]: qk *= log_2_e
                v_12 = tl.full([], 1.44269504, tl.float32)
                v_13 = v_11_0_0 * v_12
                # src[helion_flex_attention_official.py:149]: m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                _mask_to_4 = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _BLOCK_SIZE_1]), v_13, tl.full([], float('-inf'), tl.float32))
                amax = tl.cast(helion_triton_helpers.max_propagate_nan(_mask_to_4, 1), tl.float32)
                v_14 = tl.maximum(m_i_copy_0_copy_0, amax, tl.PropagateNan.ALL)
                # src[helion_flex_attention_official.py:153]: m_safe = torch.where(m_ij == float("-inf"), 0.0, m_ij)
                v_15 = tl.full([], float('-inf'), tl.float32)
                v_16 = v_14 == v_15
                scalar_tensor = tl.full([], 0.0, tl.float32)
                m_safe = tl.where(v_16, scalar_tensor, v_14)
                # src[helion_flex_attention_official.py:154]: qk = qk - m_safe[:, None]
                subscript_1 = m_safe[:, None]
                v_17 = v_13 - subscript_1
                # src[helion_flex_attention_official.py:155]: p = torch.exp2(qk)
                v_18 = libdevice.exp2(v_17)
                # src[helion_flex_attention_official.py:156]: l_ij = torch.sum(p, -1)
                _mask_to_5 = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _BLOCK_SIZE_1]), v_18, tl.full([], 0, tl.float32))
                l_ij = tl.cast(tl.sum(_mask_to_5.to(tl.float32), 1), tl.float32)
                # src[helion_flex_attention_official.py:157]: alpha = torch.exp2(m_i - m_safe)
                v_19 = m_i_copy_0_copy_0 - m_safe
                v_20 = libdevice.exp2(v_19)
                # src[helion_flex_attention_official.py:158]: m_i = m_ij
                m_i = v_14
                # src[helion_flex_attention_official.py:159]: l_i = l_i * alpha + l_ij
                v_21 = l_i_copy_0_copy_0 * v_20
                l_i = v_21 + l_ij
                # src[helion_flex_attention_official.py:160]: acc = acc * alpha[:, None]
                subscript_2 = v_20[:, None]
                v_23 = acc_copy_0_copy_0 * subscript_2
                # src[helion_flex_attention_official.py:161]: v = value[b_idx, h_kv_idx, tile_n, :]
                v = tl.load(value + (indices_6[:, None] * 512 + (0 + indices_8)[None, :] * 1), mask_1[:, None], other=0, eviction_policy='evict_last')
                # src[helion_flex_attention_official.py:162]: p = p.to(v.dtype)
                v_24 = tl.cast(v_18, tl.bfloat16)
                # src[helion_flex_attention_official.py:163]: acc = hl.dot(p, v, acc=acc)
                acc = tl.dot(tl.cast(v_24, tl.bfloat16), tl.cast(v, tl.bfloat16), acc=v_23, input_precision='tf32', out_dtype=tl.float32)
        # src[helion_flex_attention_official.py:111]: tb_idx = b_idx % table_batch
        index_2 = 0
        index_3 = 0
        # src[helion_flex_attention_official.py:166]: sparse_num_blocks = block_mask_kv_num_blocks[tb_idx, th_idx, sparse_row]
        sparse_num_blocks_1 = tl.load(block_mask_kv_num_blocks + floordiv * 1, None, eviction_policy='evict_first')
        # src[helion_flex_attention_official.py:168]: for block_idx in hl.tile(sparse_num_blocks, block_size=1):
        # src[helion_flex_attention_official.py:169]:     start_n = (
        # src[helion_flex_attention_official.py:170]:         block_mask_kv_indices[tb_idx, th_idx, sparse_row, block_idx.id]
        # src[helion_flex_attention_official.py:168-201]: ...
        for offset_7 in tl.range(0, tl.cast(sparse_num_blocks_1, tl.int32), num_stages=2, disallow_acc_multi_buffer=True):
            q_i_copy_1 = q_i
            bcast_b_copy_1 = bcast_b
            bcast_h_copy_1 = bcast_h
            bcast_m_copy_1 = bcast_m
            m_i_copy_1 = m_i
            l_i_copy_1 = l_i
            acc_copy_1 = acc
            q_i_copy_1_0 = q_i_copy_1
            bcast_b_copy_1_0 = bcast_b_copy_1
            bcast_h_copy_1_0 = bcast_h_copy_1
            bcast_m_copy_1_0 = bcast_m_copy_1
            m_i = m_i_copy_1
            l_i = l_i_copy_1
            acc = acc_copy_1
            # src[helion_flex_attention_official.py:170]: block_mask_kv_indices[tb_idx, th_idx, sparse_row, block_idx.id]
            tile_id_1 = offset_7 // _BLOCK_SIZE_7
            load_1 = tl.load(block_mask_kv_indices + (floordiv * 32 + tile_id_1 * 1), None, eviction_policy='evict_first')
            # src[helion_flex_attention_official.py:170]: block_mask_kv_indices[tb_idx, th_idx, sparse_row, block_idx.id]
            # src[helion_flex_attention_official.py:171]: * block_mask_n
            v_25 = tl.full([], 128, tl.int32)
            v_26 = tl.cast(load_1 * v_25, tl.int32)
            # src[helion_flex_attention_official.py:173]: end_n = start_n + block_mask_n
            v_27 = tl.full([], 128, tl.int32)
            v_28 = v_26 + v_27
            # src[helion_flex_attention_official.py:174]: end_N = end_n.new_full([], N)
            end_N_1 = tl.full([], 4096, tl.int32)
            # src[helion_flex_attention_official.py:175]: end_n = torch.minimum(end_n, end_N)
            v_29 = tl.minimum(v_28, end_N_1, tl.PropagateNan.ALL)
            # src[helion_flex_attention_official.py:177]: for tile_n in hl.tile(start_n, end_n, block_size=block_n):
            # src[helion_flex_attention_official.py:178]:     k = key[b_idx, h_kv_idx, tile_n, :]
            # src[helion_flex_attention_official.py:179]:     bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[None, :]
            # src[helion_flex_attention_official.py:177-201]: ...
            for offset_6 in tl.range(tl.cast(v_26, tl.int32), tl.cast(v_29, tl.int32), _BLOCK_SIZE_1, num_stages=1, disallow_acc_multi_buffer=True):
                indices_6 = offset_6 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
                mask_2 = indices_6 < v_29
                q_i_copy_1_0_copy = q_i_copy_1_0
                bcast_m_copy_1_0_copy = bcast_m_copy_1_0
                m_i_copy_1_0_copy = m_i
                l_i_copy_1_0_copy = l_i
                acc_copy_1_0_copy = acc
                q_i_copy_1_0_copy_0 = q_i_copy_1_0_copy
                bcast_m_copy_1_0_copy_0 = bcast_m_copy_1_0_copy
                m_i_copy_1_0_copy_0 = m_i_copy_1_0_copy
                l_i_copy_1_0_copy_0 = l_i_copy_1_0_copy
                acc_copy_1_0_copy_0 = acc_copy_1_0_copy
                # src[helion_flex_attention_official.py:178]: k = key[b_idx, h_kv_idx, tile_n, :]
                k_1 = tl.load(key + (indices_6[:, None] * 576 + (0 + indices_4)[None, :] * 1), mask_2[:, None] & mask_4[None, :], other=0, eviction_policy='evict_last')
                # src[helion_flex_attention_official.py:179]: bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[None, :]
                iota_3 = tl.arange(0, _BLOCK_SIZE_1)
                v_30 = tl.cast(offset_6, tl.int32)
                v_31 = iota_3 + v_30
                bcast_n = v_31[None, :]
                # src[helion_flex_attention_official.py:180]: qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                qk_2 = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
                # src[helion_flex_attention_official.py:181]: qk = hl.dot(q_i, k.T, acc=qk)
                permute_1 = tl.permute(k_1, [1, 0])
                qk_3 = tl.dot(tl.cast(q_i_copy_1_0_copy_0, tl.bfloat16), tl.cast(permute_1, tl.bfloat16), acc=qk_2, input_precision='tf32', out_dtype=tl.float32)
                # src[helion_flex_attention_official.py:183]: qk = qk * scale
                v_32 = qk_3 * scale
                # src[helion_flex_attention_official.py:184]: bcast_qk = qk
                v_32_0 = v_32
                # src[helion_flex_attention_official.py:186]: mask = block_mask_mask_mod(bcast_b, bcast_h, bcast_m, bcast_n)
                mask = tl.load(block_mask_mask_mod_closure_0 + (bcast_m_copy_1_0_copy_0 * 4097 + bcast_n * 1), mask_2[None, :], other=0, eviction_policy='evict_last')
                # src[helion_flex_attention_official.py:187]: score = torch.where(mask, score, -float("inf"))
                scalar_tensor_1 = tl.full([], float('-inf'), tl.float32)
                score_1 = tl.where(mask, v_32_0, scalar_tensor_1)
                # src[helion_flex_attention_official.py:188]: qk = score
                score_1_0 = score_1
                # src[helion_flex_attention_official.py:189]: qk *= log_2_e
                v_33 = tl.full([], 1.44269504, tl.float32)
                v_34 = score_1_0 * v_33
                # src[helion_flex_attention_official.py:190]: m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                _mask_to_6 = tl.where(tl.broadcast_to(mask_2[None, :], [_BLOCK_SIZE_0, _BLOCK_SIZE_1]), v_34, tl.full([], float('-inf'), tl.float32))
                amax_1 = tl.cast(helion_triton_helpers.max_propagate_nan(_mask_to_6, 1), tl.float32)
                v_35 = tl.maximum(m_i_copy_1_0_copy_0, amax_1, tl.PropagateNan.ALL)
                # src[helion_flex_attention_official.py:191]: m_safe = torch.where(m_ij == float("-inf"), 0.0, m_ij)
                v_36 = tl.full([], float('-inf'), tl.float32)
                v_37 = v_35 == v_36
                scalar_tensor_2 = tl.full([], 0.0, tl.float32)
                m_safe_1 = tl.where(v_37, scalar_tensor_2, v_35)
                # src[helion_flex_attention_official.py:192]: qk = qk - m_safe[:, None]
                subscript_3 = m_safe_1[:, None]
                v_38 = v_34 - subscript_3
                # src[helion_flex_attention_official.py:193]: p = torch.exp2(qk)
                v_39 = libdevice.exp2(v_38)
                # src[helion_flex_attention_official.py:194]: l_ij = torch.sum(p, -1)
                _mask_to_7 = tl.where(tl.broadcast_to(mask_2[None, :], [_BLOCK_SIZE_0, _BLOCK_SIZE_1]), v_39, tl.full([], 0, tl.float32))
                l_ij_1 = tl.cast(tl.sum(_mask_to_7.to(tl.float32), 1), tl.float32)
                # src[helion_flex_attention_official.py:195]: alpha = torch.exp2(m_i - m_safe)
                v_40 = m_i_copy_1_0_copy_0 - m_safe_1
                v_41 = libdevice.exp2(v_40)
                # src[helion_flex_attention_official.py:196]: m_i = m_ij
                m_i = v_35
                # src[helion_flex_attention_official.py:197]: l_i = l_i * alpha + l_ij
                v_42 = l_i_copy_1_0_copy_0 * v_41
                l_i = v_42 + l_ij_1
                # src[helion_flex_attention_official.py:198]: acc = acc * alpha[:, None]
                subscript_4 = v_41[:, None]
                v_44 = acc_copy_1_0_copy_0 * subscript_4
                # src[helion_flex_attention_official.py:199]: v = value[b_idx, h_kv_idx, tile_n, :]
                v_45 = tl.load(value + (indices_6[:, None] * 512 + (0 + indices_8)[None, :] * 1), mask_2[:, None], other=0)
                # src[helion_flex_attention_official.py:200]: p = p.to(v.dtype)
                v_46 = tl.cast(v_39, tl.bfloat16)
                # src[helion_flex_attention_official.py:201]: acc = hl.dot(p, v, acc=acc)
                acc = tl.dot(tl.cast(v_46, tl.bfloat16), tl.cast(v_45, tl.bfloat16), acc=v_44, input_precision='tf32', out_dtype=tl.float32)
        # src[helion_flex_attention_official.py:204]: l_i = torch.where(l_i == 0.0, 1.0, l_i)
        v_47 = tl.full([], 0.0, tl.float32)
        v_48 = l_i == v_47
        scalar_tensor_3 = tl.full([], 1.0, tl.float32)
        l_i_1 = tl.where(v_48, scalar_tensor_3, l_i)
        # src[helion_flex_attention_official.py:205]: lse_row = (m_i + torch.log2(l_i)) / log_2_e  # Convert back to natural log
        v_49 = libdevice.log2(l_i_1)
        v_50 = m_i + v_49
        v_51 = tl.full([], 0.6931471809870504, tl.float32)
        v_52 = v_50 * v_51
        # src[helion_flex_attention_official.py:206]: acc = acc / l_i[:, None]
        subscript_5 = l_i_1[:, None]
        v_53 = fast_dividef(tl.cast(acc, tl.float32), tl.cast(subscript_5, tl.float32))
        # src[helion_flex_attention_official.py:207]: out[tile_b, tile_h, tile_m, :] = acc[None, None, :, :].to(out.dtype)
        subscript_6 = v_53[None, None, :, :]
        v_54 = tl.cast(subscript_6, tl.bfloat16)
        tl.store(out + tl.broadcast_to(indices_3[None, :, None, None] * 2097152 + indices_0[None, None, :, None] * 512 + (0 + indices_8)[None, None, None, :] * 1, [_BLOCK_SIZE_2, _BLOCK_SIZE_3, _BLOCK_SIZE_0, _RDIM_SIZE_6]), v_54, None)
        # src[helion_flex_attention_official.py:208]: lse[tile_b, tile_h, tile_m] = lse_row[None, None, :]
        subscript_7 = v_52[None, None, :]
        tl.store(lse + tl.broadcast_to(indices_3[None, :, None] * 4096 + indices_0[None, None, :] * 1, [_BLOCK_SIZE_2, _BLOCK_SIZE_3, _BLOCK_SIZE_0]), subscript_7, None)

def helion_flex_attention_kernel(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask_kv_num_blocks: torch.Tensor, block_mask_kv_indices: torch.Tensor, block_mask_full_kv_num_blocks: torch.Tensor | None, block_mask_full_kv_indices: torch.Tensor | None, block_mask_mask_mod: Callable, block_mask_m: int, block_mask_n: int, scale: float, enable_gqa: bool=False, *, _launcher=_default_launcher):
    # src[helion_flex_attention_official.py:62]: B, H, M, D = query.size()
    B, H, M, D = query.size()
    # src[helion_flex_attention_official.py:63]: D = hl.specialize(D)
    D = 576
    # src[helion_flex_attention_official.py:66]: Dv = hl.specialize(value.size(-1))
    Dv = 512
    # src[helion_flex_attention_official.py:67]: Bk, Hk, N, Dk = key.size()
    Bk, Hk, N, Dk = key.size()
    # src[helion_flex_attention_official.py:68]: assert Bk == B
    assert Bk == B
    # src[helion_flex_attention_official.py:69]: assert Dk == D
    assert Dk == D
    # src[helion_flex_attention_official.py:70]: assert value.size()[:3] == key.size()[:3]
    assert value.size()[:3] == key.size()[:3]
    # src[helion_flex_attention_official.py:71]: if enable_gqa:
    # src[helion_flex_attention_official.py:72]:     assert H % Hk == 0
    # src[helion_flex_attention_official.py:73]: else:
    # src[helion_flex_attention_official.py:71-74]: ...
    if enable_gqa:
        # src[helion_flex_attention_official.py:72]: assert H % Hk == 0
        assert H % Hk == 0
    else:
        # src[helion_flex_attention_official.py:74]: assert Hk == H
        assert Hk == H
    # src[helion_flex_attention_official.py:86]: table_batch, table_heads, _ = block_mask_kv_num_blocks.size()
    table_batch, table_heads, _ = block_mask_kv_num_blocks.size()
    # src[helion_flex_attention_official.py:90]: if Dv == D:
    # src[helion_flex_attention_official.py:91]:     out = torch.empty_like(query)
    # src[helion_flex_attention_official.py:92]: else:
    # src[helion_flex_attention_official.py:90-93]: ...
    if Dv == D:
        # src[helion_flex_attention_official.py:91]: out = torch.empty_like(query)
        out = torch.empty_like(query)
    else:
        # src[helion_flex_attention_official.py:93]: out = torch.empty((B, H, M, Dv), dtype=query.dtype, device=query.device)
        out = torch.empty((B, H, M, Dv), dtype=query.dtype, device=query.device)
    # src[helion_flex_attention_official.py:94]: lse = torch.empty((B, H, M), dtype=torch.float32, device=out.device)
    lse = torch.empty((B, H, M), dtype=torch.float32, device=out.device)
    # src[helion_flex_attention_official.py:98]: assert (block_mask_full_kv_indices is None) == (
    # src[helion_flex_attention_official.py:99]:     block_mask_full_kv_num_blocks is None
    # src[helion_flex_attention_official.py:100]: )
    assert (block_mask_full_kv_indices is None) == (block_mask_full_kv_num_blocks is None)
    # src[helion_flex_attention_official.py:101]: for tile_b, tile_h, tile_m in hl.tile([B, H, M], block_size=[1, 1, block_m]):
    _NUM_SM = helion.runtime.get_num_sm(query.device)
    _RDIM_SIZE_4 = 1024
    _RDIM_SIZE_6 = 512
    # src[helion_flex_attention_official.py:101]: for tile_b, tile_h, tile_m in hl.tile([B, H, M], block_size=[1, 1, block_m]):
    # src[helion_flex_attention_official.py:102]:     m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
    # src[helion_flex_attention_official.py:103]:     l_i = torch.full_like(m_i, 1.0)
    # src[helion_flex_attention_official.py:101-208]: ...
    _launcher(_helion_helion_flex_attention_kernel, (_NUM_SM * 16,), query, block_mask_full_kv_num_blocks, block_mask_full_kv_indices, key, value, block_mask_kv_num_blocks, block_mask_kv_indices, block_mask_mask_mod.__closure__[0].cell_contents, out, lse, scale, _NUM_SM, _RDIM_SIZE_4, _RDIM_SIZE_6, num_warps=8, num_stages=8, maxnreg=256)
    # src[helion_flex_attention_official.py:209]: return out, lse
    return (out, lse)