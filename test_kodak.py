"""
Batch test on the 24 standard Kodak images.

Downloads images to kodak_images/ (cached for future runs),
runs encode + decode, verifies lossless reconstruction,
and prints a summary table. No decoded images are saved.

Usage:
    python test_kodak.py
"""

import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
from tabulate import tabulate

from python.codec import encode, decode

KODAK_DIR = os.path.join(os.path.dirname(__file__), "kodak_images")
KODAK_URL = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
NUM_IMAGES = 24


def download_kodak_images():
    os.makedirs(KODAK_DIR, exist_ok=True)
    paths = []
    for i in range(1, NUM_IMAGES + 1):
        filename = f"kodim{i:02d}.png"
        filepath = os.path.join(KODAK_DIR, filename)
        if not os.path.exists(filepath):
            url = KODAK_URL.format(i)
            print(f"  Downloading {filename} ...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(url, filepath)
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")
                continue
        paths.append(filepath)
    return paths


def test_single_image(filepath):
    img = Image.open(filepath).convert("RGB")
    image = np.array(img, dtype=np.uint8)
    h, w, _ = image.shape
    num_pixels = h * w
    raw_bits = num_pixels * 24

    t0 = time.perf_counter()
    encoded = encode(image)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    reconstructed = decode(encoded)
    t_dec = time.perf_counter() - t0

    diff = image.astype(np.int32) - reconstructed.astype(np.int32)
    max_err = int(np.max(np.abs(diff)))
    lossless = max_err == 0

    comp_bits = encoded["total_bits"]
    bpp = comp_bits / num_pixels
    ratio = raw_bits / comp_bits if comp_bits > 0 else float("inf")
    savings = (1.0 - comp_bits / raw_bits) * 100.0

    ch_bpp = {}
    for ch_name in ("Y", "Co", "Cg"):
        ch = encoded["compressed_data"][ch_name]
        ch_bits = ch["ll"]["bits"]
        for level in ch["subbands"]:
            for sub in ("LH", "HL", "HH"):
                ch_bits += level[sub]["bits"]
        ch_bpp[ch_name] = ch_bits / num_pixels

    return {
        "name": os.path.basename(filepath),
        "dims": f"{w}x{h}",
        "bpp": bpp,
        "ratio": ratio,
        "savings": savings,
        "y_bpp": ch_bpp["Y"],
        "co_bpp": ch_bpp["Co"],
        "cg_bpp": ch_bpp["Cg"],
        "enc_time": t_enc,
        "dec_time": t_dec,
        "lossless": lossless,
    }


def main():
    print("=" * 78)
    print("  KODAK 24-IMAGE LOSSLESS COMPRESSION BENCHMARK")
    print("  CDF 5/3 Integer Wavelet + Adaptive Golomb-Rice")
    print("=" * 78)
    print()

    print("  Checking / downloading Kodak images ...")
    paths = download_kodak_images()
    if not paths:
        print("  ERROR: No images available. Check your internet connection.")
        sys.exit(1)
    print(f"  {len(paths)} images ready.\n")

    results = []
    for i, path in enumerate(paths, 1):
        name = os.path.basename(path)
        print(f"  [{i:2d}/{NUM_IMAGES}] {name} ...", end=" ", flush=True)
        r = test_single_image(path)
        status = "PASS" if r["lossless"] else "FAIL"
        print(f"bpp={r['bpp']:.3f}  ratio={r['ratio']:.3f}:1  "
              f"enc={r['enc_time']:.2f}s  dec={r['dec_time']:.2f}s  [{status}]")
        results.append(r)

    all_pass = all(r["lossless"] for r in results)
    n = len(results)

    avg_bpp = sum(r["bpp"] for r in results) / n
    avg_ratio = sum(r["ratio"] for r in results) / n
    avg_savings = sum(r["savings"] for r in results) / n
    avg_y = sum(r["y_bpp"] for r in results) / n
    avg_co = sum(r["co_bpp"] for r in results) / n
    avg_cg = sum(r["cg_bpp"] for r in results) / n
    total_enc = sum(r["enc_time"] for r in results)
    total_dec = sum(r["dec_time"] for r in results)

    print()
    print("=" * 78)
    print("  DETAILED RESULTS")
    print("=" * 78)
    print()

    table = []
    for r in results:
        table.append([
            r["name"],
            r["dims"],
            f"{r['bpp']:.3f}",
            f"{r['ratio']:.3f}",
            f"{r['savings']:.1f}%",
            f"{r['y_bpp']:.3f}",
            f"{r['co_bpp']:.3f}",
            f"{r['cg_bpp']:.3f}",
            f"{r['enc_time']:.2f}",
            f"{r['dec_time']:.2f}",
            "PASS" if r["lossless"] else "FAIL",
        ])

    table.append([
        "AVERAGE", "",
        f"{avg_bpp:.3f}",
        f"{avg_ratio:.3f}",
        f"{avg_savings:.1f}%",
        f"{avg_y:.3f}",
        f"{avg_co:.3f}",
        f"{avg_cg:.3f}",
        f"{total_enc:.2f}",
        f"{total_dec:.2f}",
        "ALL PASS" if all_pass else "SOME FAIL",
    ])

    headers = ["Image", "Size", "bpp", "Ratio", "Savings",
               "Y bpp", "Co bpp", "Cg bpp", "Enc(s)", "Dec(s)", "Lossless"]
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print()

    print("=" * 78)
    print(f"  Average bpp    : {avg_bpp:.3f}")
    print(f"  Average ratio  : {avg_ratio:.3f}:1")
    print(f"  Average savings: {avg_savings:.1f}%")
    print(f"  Total encode   : {total_enc:.2f}s")
    print(f"  Total decode   : {total_dec:.2f}s")
    print(f"  Lossless       : {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
