"""
Full lossless image compression codec — enhanced pipeline.

Pipeline: RGB -> YCoCg-R -> CDF 5/3 Wavelet -> DPCM(LL) -> Adaptive Golomb-Rice

Enhancements:
  - MED-based DPCM on final LL subband
  - 6-neighbor weighted spatial context
  - Parent DETAIL context for non-coarsest detail subbands (wavelet tree)
"""

import numpy as np

from python.color_transform import rgb_to_ycocg_r, ycocg_r_to_rgb
from python.wavelet_transform import multilevel_forward, multilevel_inverse
from python.entropy_coder import (
    encode_subband, decode_subband,
    encode_subband_with_parent, decode_subband_with_parent,
    dpcm_forward, dpcm_inverse,
)


WAVELET_LEVELS = 3


def encode(image: np.ndarray) -> dict:
    assert image.ndim == 3 and image.shape[2] == 3, "Expected HxWx3 RGB image"
    h, w, _ = image.shape

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    y, co, cg = rgb_to_ycocg_r(r, g, b)
    channels = {"Y": y, "Co": co, "Cg": cg}

    compressed_data = {}
    total_compressed_bits = 0
    header_bits = 32 + 8

    for name, ch in channels.items():
        final_ll, subbands = multilevel_forward(ch, levels=WAVELET_LEVELS)
        # subbands[0] = finest (LH1,HL1,HH1), subbands[-1] = coarsest

        ch_data = {"shape": ch.shape, "levels": WAVELET_LEVELS}

        # --- Encode LL with DPCM + spatial context ---
        ll_residuals = dpcm_forward(final_ll)
        bits_ll, packed_ll = encode_subband(ll_residuals)
        ch_data["ll"] = {
            "shape": final_ll.shape,
            "bits": bits_ll,
            "packed": packed_ll,
        }
        total_compressed_bits += bits_ll
        header_bits += 32

        # --- Encode detail subbands ---
        # Coarsest details: spatial context only (no parent detail exists)
        # Finer details: spatial + parent detail from the same subband type
        #   at the next coarser level (wavelet tree structure)

        ch_subbands = []
        for level_idx, (LH, HL, HH) in enumerate(subbands):
            level_data = {}

            if level_idx == len(subbands) - 1:
                # Coarsest detail level — no parent, spatial only
                for sub_name, sub_arr in [("LH", LH), ("HL", HL), ("HH", HH)]:
                    bits, packed = encode_subband(sub_arr)
                    level_data[sub_name] = {
                        "shape": sub_arr.shape,
                        "bits": bits,
                        "packed": packed,
                    }
                    total_compressed_bits += bits
                    header_bits += 32
            else:
                # Finer level — use parent detail from coarser level
                parent_LH, parent_HL, parent_HH = subbands[level_idx + 1]
                parents = {"LH": parent_LH, "HL": parent_HL, "HH": parent_HH}

                for sub_name, sub_arr in [("LH", LH), ("HL", HL), ("HH", HH)]:
                    bits, packed = encode_subband_with_parent(
                        sub_arr, parents[sub_name]
                    )
                    level_data[sub_name] = {
                        "shape": sub_arr.shape,
                        "bits": bits,
                        "packed": packed,
                    }
                    total_compressed_bits += bits
                    header_bits += 32

            ch_subbands.append(level_data)

        ch_data["subbands"] = ch_subbands
        compressed_data[name] = ch_data

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


def decode(encoded: dict) -> np.ndarray:
    h, w = encoded["image_shape"]
    compressed_data = encoded["compressed_data"]

    channels_decoded = {}
    for name in ("Y", "Co", "Cg"):
        ch = compressed_data[name]
        num_levels = ch["levels"]
        detail_data = ch["subbands"]

        # --- Decode LL with inverse DPCM ---
        ll_info = ch["ll"]
        ll_residuals = decode_subband(
            ll_info["packed"], ll_info["bits"], ll_info["shape"]
        )
        final_ll = dpcm_inverse(ll_residuals)

        # --- Decode detail subbands coarse-to-fine ---
        # Coarsest first (spatial context only), then finer levels
        # with parent detail context from the already-decoded coarser level.

        decoded_details = [None] * num_levels

        # 1) Coarsest detail level — spatial only
        coarsest = num_levels - 1
        level = detail_data[coarsest]
        decoded_subs = []
        for sub_name in ("LH", "HL", "HH"):
            info = level[sub_name]
            sub_arr = decode_subband(
                info["packed"], info["bits"], info["shape"]
            )
            decoded_subs.append(sub_arr)
        decoded_details[coarsest] = tuple(decoded_subs)

        # 2) Finer levels — use parent detail from just-decoded coarser level
        for lvl in range(coarsest - 1, -1, -1):
            parent_details = decoded_details[lvl + 1]
            parent_map = {"LH": parent_details[0],
                          "HL": parent_details[1],
                          "HH": parent_details[2]}

            level = detail_data[lvl]
            decoded_subs = []
            for sub_name in ("LH", "HL", "HH"):
                info = level[sub_name]
                sub_arr = decode_subband_with_parent(
                    info["packed"], info["bits"], info["shape"],
                    parent_map[sub_name]
                )
                decoded_subs.append(sub_arr)
            decoded_details[lvl] = tuple(decoded_subs)

        # --- Inverse wavelet ---
        subbands_for_inverse = [(d[0], d[1], d[2]) for d in decoded_details]
        reconstructed = multilevel_inverse(final_ll, subbands_for_inverse)
        channels_decoded[name] = reconstructed

    r, g, b = ycocg_r_to_rgb(
        channels_decoded["Y"],
        channels_decoded["Co"],
        channels_decoded["Cg"]
    )
    return np.stack([r, g, b], axis=2)
