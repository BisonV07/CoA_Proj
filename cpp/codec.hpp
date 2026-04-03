#pragma once
#include "matrix.hpp"
#include "color_transform.hpp"
#include "wavelet_transform.hpp"
#include "entropy_coder.hpp"
#include <array>
#include <string>

static constexpr int WAVELET_LEVELS = 3;

// -----------------------------------------------------------------------
// Encoded-image container
// -----------------------------------------------------------------------

struct LevelEncoded {
    SubbandEncoded lh, hl, hh;
};

struct ChannelEncoded {
    int rows, cols, levels;
    SubbandEncoded ll;
    std::vector<LevelEncoded> subbands;
};

struct EncodedImage {
    int height = 0, width = 0;
    int64_t raw_bits = 0;
    int64_t compressed_bits = 0;
    int64_t header_bits = 0;
    int64_t total_bits = 0;
    std::array<ChannelEncoded, 3> channels; // Y, Co, Cg

    int64_t channel_bits(int ch) const {
        int64_t bits = channels[ch].ll.total_bits;
        for (auto& lv : channels[ch].subbands)
            bits += lv.lh.total_bits + lv.hl.total_bits + lv.hh.total_bits;
        return bits;
    }
};

// -----------------------------------------------------------------------
// Encode
// -----------------------------------------------------------------------

inline EncodedImage encode(const uint8_t* rgb, int height, int width) {
    // Split into R, G, B matrices
    Matrix r_ch(height, width), g_ch(height, width), b_ch(height, width);
    for (int i = 0; i < height * width; i++) {
        r_ch.data[i] = rgb[i * 3 + 0];
        g_ch.data[i] = rgb[i * 3 + 1];
        b_ch.data[i] = rgb[i * 3 + 2];
    }

    // Color transform
    Matrix y, co, cg;
    rgb_to_ycocg_r(r_ch, g_ch, b_ch, y, co, cg);
    Matrix ch_arr[3] = {std::move(y), std::move(co), std::move(cg)};

    EncodedImage enc;
    enc.height = height;
    enc.width  = width;
    enc.raw_bits = (int64_t)height * width * 24;
    enc.header_bits = 32 + 8; // image dims + levels

    for (int ci = 0; ci < 3; ci++) {
        auto& ch = enc.channels[ci];
        ch.rows = ch_arr[ci].rows;
        ch.cols = ch_arr[ci].cols;
        ch.levels = WAVELET_LEVELS;

        WaveletDecomp decomp = multilevel_forward(ch_arr[ci], WAVELET_LEVELS);

        // LL: DPCM then encode
        Matrix ll_res = dpcm_forward(decomp.ll);
        ch.ll = encode_subband(ll_res);
        enc.compressed_bits += ch.ll.total_bits;
        enc.header_bits += 32;

        // Detail subbands
        ch.subbands.resize(WAVELET_LEVELS);
        for (int li = 0; li < WAVELET_LEVELS; li++) {
            auto& lv = ch.subbands[li];
            const auto& wlv = decomp.levels[li];

            if (li == WAVELET_LEVELS - 1) {
                // Coarsest detail — spatial context only
                lv.lh = encode_subband(wlv.lh);
                lv.hl = encode_subband(wlv.hl);
                lv.hh = encode_subband(wlv.hh);
            } else {
                // Finer level — use parent from coarser level
                const auto& parent = decomp.levels[li + 1];
                lv.lh = encode_subband_with_parent(wlv.lh, parent.lh);
                lv.hl = encode_subband_with_parent(wlv.hl, parent.hl);
                lv.hh = encode_subband_with_parent(wlv.hh, parent.hh);
            }

            enc.compressed_bits += lv.lh.total_bits + lv.hl.total_bits + lv.hh.total_bits;
            enc.header_bits += 32 * 3;
        }
    }

    enc.total_bits = enc.compressed_bits + enc.header_bits;
    return enc;
}

// -----------------------------------------------------------------------
// Decode
// -----------------------------------------------------------------------

inline std::vector<uint8_t> decode(const EncodedImage& enc) {
    int height = enc.height, width = enc.width;
    Matrix channels_dec[3];

    for (int ci = 0; ci < 3; ci++) {
        const auto& ch = enc.channels[ci];

        // Decode LL
        Matrix ll_res = decode_subband(ch.ll);
        Matrix final_ll = dpcm_inverse(ll_res);

        // Decode detail subbands coarse-to-fine
        std::vector<WaveletLevel> decoded_levels(ch.levels);

        // 1) Coarsest — spatial only
        int coarsest = ch.levels - 1;
        const auto& clv = ch.subbands[coarsest];
        decoded_levels[coarsest].lh = decode_subband(clv.lh);
        decoded_levels[coarsest].hl = decode_subband(clv.hl);
        decoded_levels[coarsest].hh = decode_subband(clv.hh);

        // 2) Finer levels — use parent from just-decoded coarser level
        for (int li = coarsest - 1; li >= 0; li--) {
            const auto& parent = decoded_levels[li + 1];
            const auto& elv = ch.subbands[li];
            decoded_levels[li].lh = decode_subband_with_parent(elv.lh, parent.lh);
            decoded_levels[li].hl = decode_subband_with_parent(elv.hl, parent.hl);
            decoded_levels[li].hh = decode_subband_with_parent(elv.hh, parent.hh);
        }

        // Inverse wavelet
        WaveletDecomp decomp;
        decomp.ll = std::move(final_ll);
        decomp.levels = std::move(decoded_levels);
        channels_dec[ci] = multilevel_inverse(decomp);
    }

    // Inverse color transform
    Matrix r_ch, g_ch, b_ch;
    ycocg_r_to_rgb(channels_dec[0], channels_dec[1], channels_dec[2],
                   r_ch, g_ch, b_ch);

    // Pack to interleaved RGB uint8
    std::vector<uint8_t> out(height * width * 3);
    for (int i = 0; i < height * width; i++) {
        out[i * 3 + 0] = (uint8_t)r_ch.data[i];
        out[i * 3 + 1] = (uint8_t)g_ch.data[i];
        out[i * 3 + 2] = (uint8_t)b_ch.data[i];
    }
    return out;
}
