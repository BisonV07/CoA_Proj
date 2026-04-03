"""
Lossless Wavelet Image Compression -- Main Entry Point

Usage:
    python compress.py <path_to_image> [--max-dim N]

Options:
    --max-dim N    Resize the image so its largest dimension is N pixels
                   (preserves aspect ratio). Useful for quick testing on
                   large images. Omit for full-resolution processing.

Compresses the image losslessly, saves the decoded output, and reports:
  - Original vs compressed file sizes
  - Compression ratio and bpp
  - Pixel-level error (must be 0% for lossless)
  - Per-channel breakdown
"""

import sys
import os
import time
import argparse

import numpy as np
from PIL import Image
from tabulate import tabulate

from python.codec import encode, decode


def load_image(path: str, max_dim: int = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if max_dim and max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        # Ensure even dimensions for clean wavelet decomposition
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized to      : {new_w} x {new_h} (--max-dim {max_dim})")
    return np.array(img, dtype=np.uint8)


def run_compression(image_path: str, max_dim: int = None):
    print("=" * 70)
    print("  LOSSLESS WAVELET IMAGE COMPRESSION")
    print("  CDF 5/3 Integer Wavelet + Adaptive Golomb-Rice")
    print("=" * 70)
    print()

    # --- Load ---
    image = load_image(image_path, max_dim)
    h, w, c = image.shape
    num_pixels = h * w
    raw_bytes = num_pixels * 3
    raw_bits  = raw_bytes * 8
    file_size_on_disk = os.path.getsize(image_path)

    print(f"  Input file      : {os.path.basename(image_path)}")
    print(f"  Dimensions      : {w} x {h}  ({num_pixels:,} pixels)")
    print(f"  Raw size (RGB)  : {raw_bytes:,} bytes  ({raw_bits:,} bits)")
    print(f"  File on disk    : {file_size_on_disk:,} bytes "
          f"({os.path.splitext(image_path)[1]})")
    print()

    # --- Encode ---
    print("  Encoding...", end=" ", flush=True)
    t0 = time.perf_counter()
    encoded = encode(image)
    t_enc = time.perf_counter() - t0
    print(f"done in {t_enc:.2f}s")

    # --- Decode ---
    print("  Decoding...", end=" ", flush=True)
    t0 = time.perf_counter()
    reconstructed = decode(encoded)
    t_dec = time.perf_counter() - t0
    print(f"done in {t_dec:.2f}s")

    # --- Save decoded output ---
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.dirname(image_path) or "."
    decoded_path = os.path.join(out_dir, f"{base}_decoded.png")
    Image.fromarray(reconstructed.astype(np.uint8)).save(decoded_path)
    decoded_size = os.path.getsize(decoded_path)
    print(f"  Decoded saved to: {decoded_path} ({decoded_size:,} bytes)")
    print()

    # --- Lossless verification ---
    diff = image.astype(np.int32) - reconstructed.astype(np.int32)
    max_abs_err = int(np.max(np.abs(diff)))
    mean_abs_err = float(np.mean(np.abs(diff)))
    num_wrong_pixels = int(np.count_nonzero(np.any(diff != 0, axis=2)))
    pct_error = (num_wrong_pixels / num_pixels) * 100.0

    # --- Compression stats ---
    comp_bits   = encoded["total_bits"]
    comp_bytes  = (comp_bits + 7) // 8
    coeff_bits  = encoded["compressed_bits"]
    header_bits = encoded["header_bits"]

    bpp = comp_bits / num_pixels
    ratio = raw_bits / comp_bits if comp_bits > 0 else float("inf")
    savings_pct = (1.0 - comp_bits / raw_bits) * 100.0

    # --- Display results ---
    print("-" * 70)
    print("  COMPRESSION RESULTS")
    print("-" * 70)

    size_table = [
        ["Raw image (H x W x 3)", f"{raw_bytes:,} bytes", f"{raw_bits:,} bits"],
        ["Coefficient data", f"{(coeff_bits+7)//8:,} bytes", f"{coeff_bits:,} bits"],
        ["Header overhead", f"{(header_bits+7)//8:,} bytes", f"{header_bits:,} bits"],
        ["Total compressed", f"{comp_bytes:,} bytes", f"{comp_bits:,} bits"],
    ]
    print()
    print(tabulate(size_table, headers=["Component", "Size (bytes)", "Size (bits)"],
                   tablefmt="grid"))
    print()

    metrics_table = [
        ["Compression Ratio", f"{ratio:.3f} : 1"],
        ["Bits Per Pixel (bpp)", f"{bpp:.3f}"],
        ["Space Savings", f"{savings_pct:.2f}%"],
        ["Encode Time", f"{t_enc:.3f}s"],
        ["Decode Time", f"{t_dec:.3f}s"],
    ]
    print(tabulate(metrics_table, headers=["Metric", "Value"],
                   tablefmt="grid"))
    print()

    # --- Error report ---
    print("-" * 70)
    print("  LOSSLESS VERIFICATION")
    print("-" * 70)
    print()

    err_table = [
        ["Max absolute error", f"{max_abs_err}"],
        ["Mean absolute error", f"{mean_abs_err:.6f}"],
        ["Wrong pixels", f"{num_wrong_pixels:,} / {num_pixels:,}"],
        ["Pixel error %", f"{pct_error:.4f}%"],
    ]
    print(tabulate(err_table, headers=["Check", "Result"],
                   tablefmt="grid"))
    print()

    if max_abs_err == 0:
        print("  >> PASS: Reconstruction is bit-exact (lossless). <<")
    else:
        print("  >> FAIL: Reconstruction has errors! <<")

    # --- Per-channel breakdown ---
    print()
    print("-" * 70)
    print("  PER-CHANNEL BREAKDOWN")
    print("-" * 70)
    print()

    ch_rows = []
    for ch_name in ("Y", "Co", "Cg"):
        ch = encoded["compressed_data"][ch_name]
        ch_bits = ch["ll"]["bits"]
        for level in ch["subbands"]:
            for sub in ("LH", "HL", "HH"):
                ch_bits += level[sub]["bits"]
        ch_bpp = ch_bits / num_pixels
        ch_ratio = (num_pixels * 8) / ch_bits if ch_bits > 0 else float("inf")
        ch_rows.append([ch_name, f"{ch_bits:,}", f"{ch_bpp:.3f}", f"{ch_ratio:.3f}:1"])

    print(tabulate(ch_rows, headers=["Channel", "Total bits", "bpp", "Ratio"],
                   tablefmt="grid"))
    print()

    # --- Target check ---
    print("=" * 70)
    if ratio >= 2.0:
        print(f"  TARGET MET: {ratio:.3f}:1 >= 2:1")
    else:
        print(f"  TARGET NOT MET: {ratio:.3f}:1 < 2:1")
        print(f"  (Need {((raw_bits / 2) / 8):,.0f} bytes or less; "
              f"got {comp_bytes:,} bytes)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lossless wavelet image compression")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Resize largest dimension to N pixels for quick testing")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: file not found: {args.image}")
        sys.exit(1)

    run_compression(args.image, args.max_dim)
