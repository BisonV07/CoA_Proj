"""
Full lossless image compression codec — enhanced pipeline.

Pipeline: RGB -> YCoCg-R -> CDF 5/3 Wavelet -> DPCM(LL) -> Adaptive Golomb-Rice

Enhancements:
  - MED-based DPCM on final LL subband
  - 6-neighbor weighted spatial context
  - Parent DETAIL context for non-coarsest detail subbands (wavelet tree)
  - Cross-channel Y context for Co/Cg subbands (luma-chroma correlation)
"""

import numpy as np

from python.color_transform import rgb_to_ycocg_r, ycocg_r_to_rgb
from python.wavelet_transform import multilevel_forward, multilevel_inverse
from python.entropy_coder import (
    encode_subband, decode_subband,
    encode_subband_with_parent, decode_subband_with_parent,
    encode_subband_with_cross, decode_subband_with_cross,
    encode_subband_with_parent_and_cross, decode_subband_with_parent_and_cross,
    dpcm_forward, dpcm_inverse,
)


WAVELET_LEVELS = 4


def _encode_channel(ch, levels, y_cross=None):
    """Encode one channel, optionally using cross-channel Y context.

    y_cross: if provided, a dict with keys 'll_residuals' and 'subbands'
             containing Y's wavelet data for cross-channel context.
    """
    final_ll, subbands = multilevel_forward(ch, levels=levels)

    ch_data = {"shape": ch.shape, "levels": levels}
    total_bits = 0
    header_bits = 0

    # --- Encode LL with DPCM ---
    ll_residuals = dpcm_forward(final_ll)

    if y_cross is not None:
        bits_ll, packed_ll = encode_subband_with_cross(
            ll_residuals, y_cross["ll_residuals"])
    else:
        bits_ll, packed_ll = encode_subband(ll_residuals)

    ch_data["ll"] = {
        "shape": final_ll.shape,
        "bits": bits_ll,
        "packed": packed_ll,
    }
    total_bits += bits_ll
    header_bits += 32

    # --- Encode detail subbands ---
    ch_subbands = []
    for level_idx, (LH, HL, HH) in enumerate(subbands):
        level_data = {}
        is_coarsest = (level_idx == len(subbands) - 1)

        for sub_name, sub_arr in [("LH", LH), ("HL", HL), ("HH", HH)]:
            sub_idx = {"LH": 0, "HL": 1, "HH": 2}[sub_name]

            if y_cross is not None:
                y_sub = y_cross["subbands"][level_idx][sub_idx]

                if is_coarsest:
                    bits, packed = encode_subband_with_cross(sub_arr, y_sub)
                else:
                    parent_sub = subbands[level_idx + 1][sub_idx]
                    bits, packed = encode_subband_with_parent_and_cross(
                        sub_arr, parent_sub, y_sub)
            else:
                if is_coarsest:
                    bits, packed = encode_subband(sub_arr)
                else:
                    parent_sub = subbands[level_idx + 1][sub_idx]
                    bits, packed = encode_subband_with_parent(
                        sub_arr, parent_sub)

            level_data[sub_name] = {
                "shape": sub_arr.shape,
                "bits": bits,
                "packed": packed,
            }
            total_bits += bits
            header_bits += 32

        ch_subbands.append(level_data)

    ch_data["subbands"] = ch_subbands
    return ch_data, total_bits, header_bits, ll_residuals, subbands


def encode(image: np.ndarray) -> dict:
    assert image.ndim == 3 and image.shape[2] == 3, "Expected HxWx3 RGB image"
    h, w, _ = image.shape

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    y_ch, co_ch, cg_ch = rgb_to_ycocg_r(r, g, b)

    compressed_data = {}
    total_compressed_bits = 0
    header_bits = 32 + 8

    # 1) Encode Y first (no cross-channel context)
    y_data, y_bits, y_hdr, y_ll_res, y_subs = _encode_channel(
        y_ch, WAVELET_LEVELS)
    compressed_data["Y"] = y_data
    total_compressed_bits += y_bits
    header_bits += y_hdr

    # Build cross-channel reference from Y's wavelet decomposition
    y_cross = {
        "ll_residuals": y_ll_res,
        "subbands": y_subs,  # list of (LH, HL, HH) tuples per level
    }

    # 2) Encode Co with Y cross-channel context
    co_data, co_bits, co_hdr, _, _ = _encode_channel(
        co_ch, WAVELET_LEVELS, y_cross=y_cross)
    compressed_data["Co"] = co_data
    total_compressed_bits += co_bits
    header_bits += co_hdr

    # 3) Encode Cg with Y cross-channel context
    cg_data, cg_bits, cg_hdr, _, _ = _encode_channel(
        cg_ch, WAVELET_LEVELS, y_cross=y_cross)
    compressed_data["Cg"] = cg_data
    total_compressed_bits += cg_bits
    header_bits += cg_hdr

    total_bits = total_compressed_bits + header_bits
    raw_bits = h * w * 24

    return {
        "compressed_data": compressed_data,
        "image_shape": (h, w),
        "raw_bits": raw_bits,
        "compressed_bits": total_compressed_bits,
        "header_bits": header_bits,
        "total_bits": total_bits,
    }


def _decode_channel(ch_compressed, y_cross=None):
    """Decode one channel, optionally using cross-channel Y context.

    y_cross: if provided, a dict with keys 'll_residuals' and 'subbands'
             containing Y's decoded wavelet data for cross-channel context.
    """
    num_levels = ch_compressed["levels"]
    detail_data = ch_compressed["subbands"]

    # --- Decode LL ---
    ll_info = ch_compressed["ll"]
    if y_cross is not None:
        ll_residuals = decode_subband_with_cross(
            ll_info["packed"], ll_info["bits"], ll_info["shape"],
            y_cross["ll_residuals"])
    else:
        ll_residuals = decode_subband(
            ll_info["packed"], ll_info["bits"], ll_info["shape"])
    final_ll = dpcm_inverse(ll_residuals)

    # --- Decode detail subbands coarse-to-fine ---
    decoded_details = [None] * num_levels
    coarsest = num_levels - 1

    # Coarsest detail level
    level = detail_data[coarsest]
    decoded_subs = []
    for si, sub_name in enumerate(("LH", "HL", "HH")):
        info = level[sub_name]
        if y_cross is not None:
            y_sub = y_cross["subbands"][coarsest][si]
            sub_arr = decode_subband_with_cross(
                info["packed"], info["bits"], info["shape"], y_sub)
        else:
            sub_arr = decode_subband(
                info["packed"], info["bits"], info["shape"])
        decoded_subs.append(sub_arr)
    decoded_details[coarsest] = tuple(decoded_subs)

    # Finer levels — parent + optional cross-channel
    for lvl in range(coarsest - 1, -1, -1):
        parent_details = decoded_details[lvl + 1]
        parent_map = {"LH": parent_details[0],
                      "HL": parent_details[1],
                      "HH": parent_details[2]}

        level = detail_data[lvl]
        decoded_subs = []
        for si, sub_name in enumerate(("LH", "HL", "HH")):
            info = level[sub_name]
            if y_cross is not None:
                y_sub = y_cross["subbands"][lvl][si]
                sub_arr = decode_subband_with_parent_and_cross(
                    info["packed"], info["bits"], info["shape"],
                    parent_map[sub_name], y_sub)
            else:
                sub_arr = decode_subband_with_parent(
                    info["packed"], info["bits"], info["shape"],
                    parent_map[sub_name])
            decoded_subs.append(sub_arr)
        decoded_details[lvl] = tuple(decoded_subs)

    # --- Inverse wavelet ---
    subbands_for_inverse = [(d[0], d[1], d[2]) for d in decoded_details]
    reconstructed = multilevel_inverse(final_ll, subbands_for_inverse)
    return reconstructed, ll_residuals, decoded_details


def decode(encoded: dict) -> np.ndarray:
    h, w = encoded["image_shape"]
    compressed_data = encoded["compressed_data"]

    # 1) Decode Y first (no cross-channel context)
    y_recon, y_ll_res, y_details = _decode_channel(compressed_data["Y"])

    # Build cross-channel reference from Y's decoded wavelet data
    y_cross = {
        "ll_residuals": y_ll_res,
        "subbands": y_details,  # list of (LH, HL, HH) tuples per level
    }

    # 2) Decode Co with Y cross-channel context
    co_recon, _, _ = _decode_channel(compressed_data["Co"], y_cross=y_cross)

    # 3) Decode Cg with Y cross-channel context
    cg_recon, _, _ = _decode_channel(compressed_data["Cg"], y_cross=y_cross)

    r, g, b = ycocg_r_to_rgb(y_recon, co_recon, cg_recon)
    return np.stack([r, g, b], axis=2)
