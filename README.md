# Lossless Wavelet Image Compression

A lossless image compression codec using the **CDF 5/3 integer wavelet transform** and **multi-bucket adaptive Golomb-Rice entropy coding**. Achieves an average **2.42:1 compression ratio** (9.983 bpp) on the standard Kodak image set with **bit-exact reconstruction**.

## Compression Pipeline

```
RGB → YCoCg-R → CDF 5/3 Wavelet (4 levels) → MED-DPCM (LL) → Multi-Bucket Adaptive Golomb-Rice
```

| Stage | Description |
|-------|-------------|
| **Color Transform** | Reversible YCoCg-R — integer-only arithmetic (adds + shifts), perfectly lossless |
| **Wavelet Transform** | CDF 5/3 lifting scheme, 4-level decomposition, operates entirely in int32 |
| **DPCM** | Median Edge Detector (MED) prediction on the final LL subband |
| **Context Model** | 6-neighbor spatial context + gradient-based 64-bucket adaptive model with decaying statistics, inter-scale parent context, and cross-channel Y context for chroma |
| **Entropy Coder** | Adaptive Golomb-Rice with escape coding for large values |

## Kodak Benchmark Results

Tested on all 24 standard Kodak PhotoCD images (768×512 / 512×768, 24-bit RGB):

| Metric | Value |
|--------|-------|
| Average bpp | 9.983 |
| Average ratio | 2.421 : 1 |
| Average savings | 58.4% |
| Lossless | ALL 24 PASS |
| Best (kodim03) | 8.704 bpp, 2.757 : 1 |
| Worst (kodim13) | 12.073 bpp, 1.988 : 1 |
| Images meeting 2:1 | 23 / 24 |

### Per-Channel Average bpp

| Channel | bpp |
|---------|-----|
| Y (luma) | 4.471 |
| Co (chroma-orange) | 2.833 |
| Cg (chroma-green) | 2.675 |

### Improvement History

| Metric | v1 (single k) | v2 (16-bucket) | v3 (current) | Total delta |
|--------|---------------|----------------|--------------|-------------|
| Average bpp | 10.780 | 10.094 | **9.983** | **−0.797** |
| Average ratio | 2.246 : 1 | 2.395 : 1 | **2.421 : 1** | +0.175 |
| Images ≥ 2:1 | 20 / 24 | 23 / 24 | **23 / 24** | +3 |

#### v3 Enhancements (current)

- **4-level wavelet decomposition** (was 3): deeper decorrelation, especially for luma
- **64-bucket context model** (was 16): 8 gradient levels × 8 spatial-k levels for finer adaptation
- **Bucket statistics decay**: halves running sums at count 128, preventing stale global statistics from dominating local adaptation
- **Cross-channel Y context**: uses luma wavelet coefficients as additional context for chroma (Co/Cg) entropy coding, with weight-2 influence
- **Inter-scale parent context**: wavelet-tree parent detail coefficients improve k estimation for finer-scale subbands
- **Faster warmup**: cold threshold reduced to 2 samples before bucket statistics activate

## Project Structure

```
CoA_Proj/
├── compress.py               # CLI — compress a single image
├── demo.py                   # Interactive GUI demo
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
├── kodak_images/             # Kodak test images (auto-downloaded)
└── test_images/              # Synthetic test images
```

## Getting Started

**Prerequisites:** Python 3.10+

```bash
pip install -r requirements.txt
```

### Interactive Demo (GUI)

```bash
python demo.py
```

Opens a window where you can select any image, compress it, and view side-by-side original vs decoded along with full compression statistics.

### CLI Usage

```bash
# Compress a single image
python compress.py path/to/image.png

# Quick test with resized image
python compress.py path/to/image.png --max-dim 256

# Run the full Kodak benchmark
python test_kodak.py
```
