# Lossless Wavelet Image Compression

A lossless image compression codec built in Python using the **CDF 5/3 integer wavelet transform** and **adaptive Golomb-Rice entropy coding**. Achieves an average **2.25:1 compression ratio** on the standard Kodak image set with **bit-exact reconstruction**.

## Compression Pipeline

```
RGB → YCoCg-R → CDF 5/3 Wavelet (3 levels) → MED-DPCM (LL) → Adaptive Golomb-Rice
```

| Stage | Description |
|-------|-------------|
| **Color Transform** | Reversible YCoCg-R — integer-only arithmetic (adds + shifts), perfectly lossless |
| **Wavelet Transform** | CDF 5/3 lifting scheme, fully vectorized in NumPy, operates in int32 |
| **DPCM** | Median Edge Detector (MED) prediction on the final LL subband |
| **Context Model** | 6-neighbor weighted spatial context + inter-scale parent detail context |
| **Entropy Coder** | Adaptive Golomb-Rice with escape coding for large values |

## Kodak Benchmark Results

Tested on all 24 standard Kodak PhotoCD images (768×512 / 512×768, 24-bit RGB):

| Metric | Value |
|--------|-------|
| Average bpp | 10.780 |
| Average ratio | 2.246 : 1 |
| Average savings | 55.1% |
| Lossless | ALL 24 PASS |
| Best (kodim03) | 9.090 bpp, 2.640 : 1 |
| Worst (kodim13) | 13.294 bpp, 1.805 : 1 |
| Images meeting 2:1 | 20 / 24 |

### Per-Channel Average bpp

| Channel | bpp |
|---------|-----|
| Y (luma) | 4.955 |
| Co (chroma-orange) | 2.995 |
| Cg (chroma-green) | 2.828 |

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
│   ├── entropy_coder.py      # Adaptive Golomb-Rice coder
│   └── context_model.py      # Spatial context / k-parameter estimation
├── kodak_images/             # Kodak test images (auto-downloaded)
└── test_images/              # Synthetic test images
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Compress a Single Image

```bash
python compress.py path/to/image.png
```

Use `--max-dim` to resize for quick testing:

```bash
python compress.py path/to/image.png --max-dim 256
```

The codec will encode and decode the image, save the reconstructed output as `<name>_decoded.png`, and print compression statistics with lossless verification.

### Run the Kodak Benchmark

```bash
python test_kodak.py
```

Downloads all 24 Kodak images (cached in `kodak_images/`), runs encode + decode on each, verifies lossless reconstruction, and prints a detailed summary table.

## Speed

Optimized for speed while preserving bit-identical output:

| Phase | Time (24 images) | Per Image |
|-------|-------------------|-----------|
| Encode | 20.88s | 0.87s |
| Decode | 46.19s | 1.92s |
| Total | 67.07s | 2.79s |

Key optimizations applied:
- Vectorized NumPy pre-computation of the k parameter map (replaces ~2.36M per-pixel function calls)
- Fully inlined decoder inner loop with Python lists
- Multi-bit batch operations in BitWriter / BitReader
