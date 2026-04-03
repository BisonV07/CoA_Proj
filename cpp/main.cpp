#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "codec.hpp"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <sys/stat.h>

// -----------------------------------------------------------------------
// Portable helpers (no std::filesystem needed)
// -----------------------------------------------------------------------

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool dir_exists(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return (st.st_mode & S_IFDIR) != 0;
}

static std::string path_join(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
    return dir + "\\" + file;
}

static std::string filename_from_path(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

// -----------------------------------------------------------------------
// Benchmark
// -----------------------------------------------------------------------

struct ImageResult {
    std::string name;
    int width, height;
    double bpp, ratio, savings;
    double y_bpp, co_bpp, cg_bpp;
    double enc_time, dec_time;
    bool lossless;
};

static double now_sec() {
    using clk = std::chrono::high_resolution_clock;
    static auto start = clk::now();
    return std::chrono::duration<double>(clk::now() - start).count();
}

static ImageResult test_single_image(const std::string& path) {
    ImageResult res;
    res.lossless = false;

    int w, h, ch;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!data) {
        fprintf(stderr, "  ERROR: cannot load %s\n", path.c_str());
        return res;
    }

    int num_pixels = h * w;
    int64_t raw_bits = (int64_t)num_pixels * 24;

    double t0 = now_sec();
    EncodedImage enc = encode(data, h, w);
    double t_enc = now_sec() - t0;

    t0 = now_sec();
    std::vector<uint8_t> recon = decode(enc);
    double t_dec = now_sec() - t0;

    bool lossless = true;
    for (int i = 0; i < num_pixels * 3; i++) {
        if (recon[i] != data[i]) { lossless = false; break; }
    }
    stbi_image_free(data);

    res.name     = filename_from_path(path);
    res.width    = w;
    res.height   = h;
    res.bpp      = (double)enc.total_bits / num_pixels;
    res.ratio    = (double)raw_bits / enc.total_bits;
    res.savings  = (1.0 - (double)enc.total_bits / raw_bits) * 100.0;
    res.y_bpp    = (double)enc.channel_bits(0) / num_pixels;
    res.co_bpp   = (double)enc.channel_bits(1) / num_pixels;
    res.cg_bpp   = (double)enc.channel_bits(2) / num_pixels;
    res.enc_time = t_enc;
    res.dec_time = t_dec;
    res.lossless = lossless;
    return res;
}

static void print_separator(int w, char c = '=') {
    for (int i = 0; i < w; i++) putchar(c);
    putchar('\n');
}

int main(int argc, char* argv[]) {
    std::string kodak_dir = "kodak_images";
    if (argc > 1) kodak_dir = argv[1];

    if (!dir_exists(kodak_dir)) {
        fprintf(stderr, "Error: directory '%s' not found.\n"
                "Usage: benchmark [path_to_kodak_images]\n", kodak_dir.c_str());
        return 1;
    }

    const int NUM_IMAGES = 24;
    print_separator(80);
    printf("  KODAK 24-IMAGE LOSSLESS COMPRESSION BENCHMARK  (C++)\n");
    printf("  CDF 5/3 Integer Wavelet + Adaptive Golomb-Rice\n");
    print_separator(80);
    printf("\n");

    std::vector<ImageResult> results;
    for (int i = 1; i <= NUM_IMAGES; i++) {
        char fname[64];
        snprintf(fname, sizeof(fname), "kodim%02d.png", i);
        std::string path = path_join(kodak_dir, fname);

        if (!file_exists(path)) {
            printf("  [%2d/%d] %s ... MISSING\n", i, NUM_IMAGES, fname);
            continue;
        }

        printf("  [%2d/%d] %s ...", i, NUM_IMAGES, fname);
        fflush(stdout);

        ImageResult r = test_single_image(path);
        const char* status = r.lossless ? "PASS" : "FAIL";
        printf(" bpp=%.3f  ratio=%.3f:1  enc=%.3fs  dec=%.3fs  [%s]\n",
               r.bpp, r.ratio, r.enc_time, r.dec_time, status);
        results.push_back(r);
    }

    if (results.empty()) {
        printf("\n  No images processed.\n");
        return 1;
    }

    int n = (int)results.size();
    bool all_pass = true;
    double sum_bpp = 0, sum_ratio = 0, sum_savings = 0;
    double sum_y = 0, sum_co = 0, sum_cg = 0;
    double total_enc = 0, total_dec = 0;

    for (size_t idx = 0; idx < results.size(); idx++) {
        const ImageResult& r = results[idx];
        if (!r.lossless) all_pass = false;
        sum_bpp     += r.bpp;
        sum_ratio   += r.ratio;
        sum_savings += r.savings;
        sum_y       += r.y_bpp;
        sum_co      += r.co_bpp;
        sum_cg      += r.cg_bpp;
        total_enc   += r.enc_time;
        total_dec   += r.dec_time;
    }

    double avg_bpp     = sum_bpp / n;
    double avg_ratio   = sum_ratio / n;
    double avg_savings = sum_savings / n;

    printf("\n");
    print_separator(80);
    printf("  DETAILED RESULTS\n");
    print_separator(80);
    printf("\n");
    printf("  %-13s %-8s %6s %7s %7s %6s %6s %6s %6s %6s %s\n",
           "Image", "Size", "bpp", "Ratio", "Save", "Y", "Co", "Cg",
           "Enc(s)", "Dec(s)", "OK?");
    printf("  %-13s %-8s %6s %7s %7s %6s %6s %6s %6s %6s %s\n",
           "-------------", "--------", "------", "-------", "-------",
           "------", "------", "------", "------", "------", "----");

    for (size_t idx = 0; idx < results.size(); idx++) {
        const ImageResult& r = results[idx];
        char sz[16];
        snprintf(sz, sizeof(sz), "%dx%d", r.width, r.height);
        printf("  %-13s %-8s %6.3f %6.3f %6.1f%% %6.3f %6.3f %6.3f %6.3f %6.3f %s\n",
               r.name.c_str(), sz,
               r.bpp, r.ratio, r.savings,
               r.y_bpp, r.co_bpp, r.cg_bpp,
               r.enc_time, r.dec_time,
               r.lossless ? "PASS" : "FAIL");
    }

    printf("  %-13s %-8s %6.3f %6.3f %6.1f%% %6.3f %6.3f %6.3f %6.3f %6.3f %s\n",
           "AVERAGE", "",
           avg_bpp, avg_ratio, avg_savings,
           sum_y / n, sum_co / n, sum_cg / n,
           total_enc, total_dec,
           all_pass ? "ALL PASS" : "SOME FAIL");

    printf("\n");
    print_separator(80);
    printf("  C++ BENCHMARK SUMMARY\n");
    print_separator(80);
    printf("\n");
    printf("  Average bpp     : %.3f\n", avg_bpp);
    printf("  Average ratio   : %.3f:1\n", avg_ratio);
    printf("  Average savings : %.1f%%\n", avg_savings);
    printf("  Total encode    : %.3fs  (avg %.3fs per image)\n", total_enc, total_enc / n);
    printf("  Total decode    : %.3fs  (avg %.3fs per image)\n", total_dec, total_dec / n);
    printf("  Lossless        : %s\n", all_pass ? "ALL PASS" : "SOME FAIL");

    const double PY_ENC_TIME = 20.88;
    const double PY_DEC_TIME = 46.19;
    const double PY_AVG_BPP  = 10.780;

    printf("\n");
    print_separator(80);
    printf("  PYTHON vs C++ COMPARISON\n");
    print_separator(80);
    printf("\n");
    printf("  %-20s %12s %12s %10s\n", "Metric", "Python", "C++", "Speedup");
    printf("  %-20s %12s %12s %10s\n", "--------------------",
           "------------", "------------", "----------");
    printf("  %-20s %11.3fs %11.3fs %9.1fx\n",
           "Encode (24 images)", PY_ENC_TIME, total_enc, PY_ENC_TIME / total_enc);
    printf("  %-20s %11.3fs %11.3fs %9.1fx\n",
           "Decode (24 images)", PY_DEC_TIME, total_dec, PY_DEC_TIME / total_dec);
    printf("  %-20s %11.3fs %11.3fs %9.1fx\n",
           "Total", PY_ENC_TIME + PY_DEC_TIME, total_enc + total_dec,
           (PY_ENC_TIME + PY_DEC_TIME) / (total_enc + total_dec));
    double bpp_diff = avg_bpp - PY_AVG_BPP;
    if (bpp_diff < 0) bpp_diff = -bpp_diff;
    printf("  %-20s %11.3f  %11.3f  %10s\n",
           "Avg bpp", PY_AVG_BPP, avg_bpp,
           (bpp_diff < 0.001) ? "MATCH" : "DIFF");
    printf("\n");
    print_separator(80);

    return all_pass ? 0 : 1;
}
