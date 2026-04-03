# Lossless Wavelet Image Compression

A lossless image compression codec using the **CDF 5/3 integer wavelet transform** and **multi-bucket adaptive Golomb-Rice entropy coding**. Achieves an average **2.40:1 compression ratio** (10.094 bpp) on the standard Kodak image set with **bit-exact reconstruction**.

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
├── kodak_images/             # Kodak test images (auto-downloaded)
└── test_images/              # Synthetic test images
```

## Getting Started

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
