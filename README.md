# Lossless Wavelet Image Compression

A lossless image compression codec using the **CDF 5/3 integer wavelet transform** and **multi-bucket adaptive Golomb-Rice entropy coding**. Implemented in both Python and C++. Achieves an average **2.40:1 compression ratio** (10.094 bpp) on the standard Kodak image set with **bit-exact reconstruction**.

## Compression Pipeline

```
RGB → YCoCg-R → CDF 5/3 Wavelet (3 levels) → MED-DPCM (LL) → Multi-Bucket Adaptive Golomb-Rice
```

| Stage | Description |
|-------|-------------|
| **Color Transform** | Reversible YCoCg-R — integer-only arithmetic (adds + shifts), perfectly lossless |
| **Wavelet Transform** | CDF 5/3 lifting scheme, operates entirely in int32 |
| **DPCM** | Median Edge Detector (MED) prediction on the final LL subband |
| **Context Model** | 6-neighbor spatial context + gradient-based 16-bucket adaptive model with running statistics |
| **Entropy Coder** | Adaptive Golomb-Rice with escape coding for large values |

## Kodak Benchmark Results

Tested on all 24 standard Kodak PhotoCD images (768×512 / 512×768, 24-bit RGB):

| Metric | Value |
|--------|-------|
| Average bpp | 10.094 |
| Average ratio | 2.395 : 1 |
| Average savings | 57.9% |
| Lossless | ALL 24 PASS |
| Best (kodim03) | 8.805 bpp, 2.726 : 1 |
| Worst (kodim13) | 12.261 bpp, 1.957 : 1 |
| Images meeting 2:1 | 23 / 24 |

### Per-Channel Average bpp

| Channel | bpp |
|---------|-----|
| Y (luma) | 4.535 |
| Co (chroma-orange) | 2.857 |
| Cg (chroma-green) | 2.699 |

### Improvement over Prior Version

The multi-bucket context model replaced the single per-pixel k estimation, yielding a **0.686 bpp improvement** across the full Kodak set:

| Metric | Prior | Current | Delta |
|--------|-------|---------|-------|
| Average bpp | 10.780 | 10.094 | **−0.686** |
| Average ratio | 2.246 : 1 | 2.395 : 1 | +0.149 |
| Images ≥ 2:1 | 20 / 24 | 23 / 24 | +3 |

## Python vs C++ Performance

The C++ implementation uses the prior single-k algorithm and produces **bit-identical** output at that compression level. It has not yet been updated with the multi-bucket context model.

| Metric | Python (prior) | C++ | Speedup |
|--------|----------------|-----|---------|
| Encode (24 images) | 20.88s | 1.71s | **12.2x** |
| Decode (24 images) | 46.19s | 1.60s | **28.8x** |
| Total | 67.07s | 3.31s | **20.3x** |
| Avg bpp | 10.780 | 10.780 | Identical |

## Project Structure

```
CoA_Proj/
├── compress.py               # Main CLI — compress a single image
├── test_kodak.py             # Kodak 24-image benchmark suite
├── generate_test_image.py    # Synthetic gradient image generator
├── requirements.txt
├── results_summary.txt       # Full benchmark results
├── python/
│   ├── codec.py              # Top-level encode / decode pipeline
│   ├── color_transform.py    # Reversible YCoCg-R transform
│   ├── wavelet_transform.py  # CDF 5/3 integer wavelet (lifting)
│   ├── entropy_coder.py      # Multi-bucket adaptive Golomb-Rice coder
│   └── context_model.py      # Spatial context / k-parameter estimation
├── cpp/
│   ├── CMakeLists.txt        # Build system (also supports direct g++ build)
│   ├── main.cpp              # C++ benchmark entry point
│   ├── codec.hpp             # Full encode / decode pipeline
│   ├── color_transform.hpp   # Reversible YCoCg-R transform
│   ├── wavelet_transform.hpp # CDF 5/3 integer wavelet (lifting)
│   ├── entropy_coder.hpp     # Golomb-Rice coder + BitWriter/BitReader
│   └── matrix.hpp            # 2D int32 array container
├── kodak_images/             # Kodak test images (auto-downloaded)
└── test_images/              # Synthetic test images
```

## Getting Started

### Python

**Prerequisites:** Python 3.10+

```bash
pip install -r requirements.txt

# Compress a single image
python compress.py path/to/image.png

# Run the full Kodak benchmark
python test_kodak.py
```

Use `--max-dim` to resize for quick testing:

```bash
python compress.py path/to/image.png --max-dim 256
```

### C++

**Prerequisites:** A C++ compiler with C++14 support (GCC, Clang, or MSVC)

Build with g++ (no CMake required):

```bash
g++ -std=c++14 -O2 -o cpp/benchmark.exe cpp/main.cpp -Icpp
```

Or with CMake:

```bash
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Run the benchmark:

```bash
./cpp/benchmark.exe kodak_images
```

The C++ benchmark auto-downloads `stb_image.h` via CMake, or you can place it in `cpp/` manually for direct g++ builds.
