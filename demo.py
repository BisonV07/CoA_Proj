"""
Interactive GUI demo for the lossless wavelet image codec.

Run:
    python demo.py

Opens a window where you can select an image, compress it, and view
side-by-side original vs decoded along with full compression statistics.
"""

import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk

from python.codec import encode, decode


BG = "#1e1e2e"
FG = "#cdd6f4"
ACCENT = "#89b4fa"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
SURFACE = "#313244"
SUBTEXT = "#a6adc8"
OVERLAY = "#45475a"


class DemoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lossless Wavelet Image Compression — Demo")
        self.configure(bg=BG)
        self.minsize(1060, 760)
        self._tk_images = []
        self._build_ui()
        self._center_window(1100, 800)

    def _center_window(self, w, h):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11), padding=8)
        style.configure("TLabel", background=BG, foreground=FG,
                        font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"),
                        foreground=ACCENT)
        style.configure("Sub.TLabel", font=("Segoe UI", 9),
                        foreground=SUBTEXT)
        style.configure("Pass.TLabel", font=("Segoe UI", 11, "bold"),
                        foreground=GREEN, background=BG)
        style.configure("Fail.TLabel", font=("Segoe UI", 11, "bold"),
                        foreground=RED, background=BG)

        header = ttk.Label(self, text="Lossless Wavelet Image Compression",
                           style="Header.TLabel")
        header.pack(pady=(14, 2))
        ttk.Label(
            self,
            text="CDF 5/3 Wavelet  ·  YCoCg-R  ·  Multi-Bucket Adaptive Golomb-Rice",
            style="Sub.TLabel",
        ).pack(pady=(0, 10))

        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=(0, 8))
        self.btn_select = ttk.Button(btn_frame, text="Select Image",
                                     command=self._on_select)
        self.btn_select.pack(side=tk.LEFT, padx=6)
        self.status_var = tk.StringVar(value="No image selected")
        ttk.Label(btn_frame, textvariable=self.status_var,
                  style="Sub.TLabel").pack(side=tk.LEFT, padx=10)

        outer = tk.Frame(self, bg=BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_frame = tk.Frame(canvas, bg=BG)
        self._canvas_win = canvas.create_window((0, 0), window=self.scroll_frame,
                                                 anchor="nw")
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind("<Configure>",
                     lambda e: canvas.itemconfig(self._canvas_win, width=e.width))

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.content = self.scroll_frame

        img_frame = tk.Frame(self.content, bg=BG)
        img_frame.pack(fill=tk.X)

        self.left_frame = tk.Frame(img_frame, bg=SURFACE)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                             padx=(0, 5))
        tk.Label(self.left_frame, text="Original", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(pady=(6, 2))
        self.lbl_orig = tk.Label(self.left_frame, bg=SURFACE)
        self.lbl_orig.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.right_frame = tk.Frame(img_frame, bg=SURFACE)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                              padx=(5, 0))
        tk.Label(self.right_frame, text="Decoded", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(pady=(6, 2))
        self.lbl_dec = tk.Label(self.right_frame, bg=SURFACE)
        self.lbl_dec.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.stats_frame = tk.Frame(self.content, bg=BG)
        self.stats_frame.pack(fill=tk.X, pady=(10, 0))

    # ------------------------------------------------------------------ #

    def _on_select(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.btn_select.config(state=tk.DISABLED)
        self.status_var.set(f"Processing {os.path.basename(path)}...")
        threading.Thread(target=self._process, args=(path,),
                         daemon=True).start()

    def _process(self, path):
        try:
            img = Image.open(path).convert("RGB")
            image = np.array(img, dtype=np.uint8)
            h, w, _ = image.shape
            num_pixels = h * w
            raw_bits = num_pixels * 3 * 8

            t0 = time.perf_counter()
            encoded = encode(image)
            t_enc = time.perf_counter() - t0

            t0 = time.perf_counter()
            reconstructed = decode(encoded)
            t_dec = time.perf_counter() - t0

            diff = image.astype(np.int32) - reconstructed.astype(np.int32)
            abs_diff = np.abs(diff)
            max_err = int(np.max(abs_diff))
            mean_err = float(np.mean(abs_diff))
            wrong_px = int(np.count_nonzero(np.any(diff != 0, axis=2)))

            comp_bits = encoded["total_bits"]
            coeff_bits = encoded["compressed_bits"]
            header_bits = encoded["header_bits"]
            bpp = comp_bits / num_pixels
            ratio = raw_bits / comp_bits if comp_bits > 0 else float("inf")
            savings = (1.0 - comp_bits / raw_bits) * 100.0

            channels = {}
            for ch_name in ("Y", "Co", "Cg"):
                ch = encoded["compressed_data"][ch_name]
                ch_bits = ch["ll"]["bits"]
                for level in ch["subbands"]:
                    for sub in ("LH", "HL", "HH"):
                        ch_bits += level[sub]["bits"]
                ch_bpp = ch_bits / num_pixels
                ch_ratio = (num_pixels * 8) / ch_bits if ch_bits > 0 else float("inf")
                channels[ch_name] = {
                    "bits": ch_bits,
                    "bpp": ch_bpp,
                    "ratio": ch_ratio,
                }

            dec_img = Image.fromarray(reconstructed.astype(np.uint8))

            self.after(0, self._show_results, img, dec_img, {
                "file": os.path.basename(path),
                "dims": f"{w} × {h}",
                "w": w, "h": h,
                "pixels": num_pixels,
                "bpp": bpp,
                "ratio": ratio,
                "savings": savings,
                "t_enc": t_enc,
                "t_dec": t_dec,
                "max_err": max_err,
                "mean_err": mean_err,
                "wrong_px": wrong_px,
                "lossless": max_err == 0,
                "channels": channels,
                "raw_bits": raw_bits,
                "raw_bytes": num_pixels * 3,
                "comp_bits": comp_bits,
                "comp_bytes": (comp_bits + 7) // 8,
                "coeff_bits": coeff_bits,
                "header_bits": header_bits,
            })
        except Exception as e:
            self.after(0, self._show_error, str(e))

    # ------------------------------------------------------------------ #

    def _fit_image(self, pil_img, max_w, max_h):
        w, h = pil_img.size
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        return pil_img.resize((nw, nh), Image.LANCZOS)

    def _make_section(self, parent, title):
        frame = tk.Frame(parent, bg=SURFACE, padx=12, pady=8)
        frame.pack(fill=tk.X, pady=(8, 0))
        tk.Label(frame, text=title, bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(
            fill=tk.X, pady=(0, 4))
        inner = tk.Frame(frame, bg=SURFACE)
        inner.pack(fill=tk.X)
        return inner

    def _add_metric(self, parent, label, value, row, val_fg=FG):
        tk.Label(parent, text=label, bg=SURFACE, fg=SUBTEXT, anchor="w",
                 font=("Segoe UI", 9)).grid(
            row=row, column=0, sticky="w", padx=(0, 16), pady=1)
        tk.Label(parent, text=value, bg=SURFACE, fg=val_fg, anchor="e",
                 font=("Consolas", 9, "bold")).grid(
            row=row, column=1, sticky="e", pady=1)

    def _add_table(self, parent, headers, rows):
        tbl = tk.Frame(parent, bg=SURFACE)
        tbl.pack(fill=tk.X, pady=(2, 0))
        for c, h in enumerate(headers):
            anchor = "w" if c == 0 else "e"
            tk.Label(tbl, text=h, bg=OVERLAY, fg=ACCENT, anchor=anchor,
                     font=("Segoe UI", 9, "bold"), padx=10, pady=3).grid(
                row=0, column=c, sticky="we", padx=(0, 1))
        for r, row_data in enumerate(rows, start=1):
            for c, cell in enumerate(row_data):
                anchor = "w" if c == 0 else "e"
                tk.Label(tbl, text=cell, bg=SURFACE, fg=FG, anchor=anchor,
                         font=("Consolas", 9), padx=10, pady=2).grid(
                    row=r, column=c, sticky="we")
        for c in range(len(headers)):
            tbl.columnconfigure(c, weight=1)

    # ------------------------------------------------------------------ #

    def _show_results(self, orig_pil, dec_pil, s):
        self._tk_images.clear()
        self.status_var.set(f"{s['file']}  —  {s['dims']}")
        self.btn_select.config(state=tk.NORMAL)

        max_w, max_h = 460, 300
        orig_tk = ImageTk.PhotoImage(self._fit_image(orig_pil, max_w, max_h))
        dec_tk = ImageTk.PhotoImage(self._fit_image(dec_pil, max_w, max_h))
        self._tk_images.extend([orig_tk, dec_tk])
        self.lbl_orig.config(image=orig_tk)
        self.lbl_dec.config(image=dec_tk)

        for w in self.stats_frame.winfo_children():
            w.destroy()

        # -- Image Info ------------------------------------------------ #
        sec = self._make_section(self.stats_frame, "Image Info")
        self._add_metric(sec, "File", s["file"], 0)
        self._add_metric(sec, "Dimensions", f"{s['w']} × {s['h']}", 1)
        self._add_metric(sec, "Pixels", f"{s['pixels']:,}", 2)
        self._add_metric(sec, "Color Depth", "24-bit RGB (8 bpc)", 3)

        # -- Compression Summary --------------------------------------- #
        cols_frame = tk.Frame(self.stats_frame, bg=BG)
        cols_frame.pack(fill=tk.X, pady=(8, 0))

        col_l = tk.Frame(cols_frame, bg=SURFACE, padx=12, pady=8)
        col_l.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Label(col_l, text="Compression", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(
            fill=tk.X, pady=(0, 4))
        cl = tk.Frame(col_l, bg=SURFACE)
        cl.pack(fill=tk.X)
        self._add_metric(cl, "Bits Per Pixel", f"{s['bpp']:.3f}", 0)
        self._add_metric(cl, "Compression Ratio", f"{s['ratio']:.3f} : 1", 1)
        self._add_metric(cl, "Space Savings", f"{s['savings']:.1f}%", 2)
        target_met = s["ratio"] >= 2.0
        self._add_metric(cl, "Target (2:1)",
                         "MET" if target_met else "NOT MET", 3,
                         val_fg=GREEN if target_met else RED)

        col_r = tk.Frame(cols_frame, bg=SURFACE, padx=12, pady=8)
        col_r.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        tk.Label(col_r, text="Speed", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(
            fill=tk.X, pady=(0, 4))
        cr = tk.Frame(col_r, bg=SURFACE)
        cr.pack(fill=tk.X)
        self._add_metric(cr, "Encode Time", f"{s['t_enc']:.3f} s", 0)
        self._add_metric(cr, "Decode Time", f"{s['t_dec']:.3f} s", 1)
        self._add_metric(cr, "Total Time",
                         f"{s['t_enc'] + s['t_dec']:.3f} s", 2)
        px_per_sec = s["pixels"] / (s["t_enc"] + s["t_dec"])
        self._add_metric(cr, "Throughput", f"{px_per_sec / 1e6:.2f} Mpx/s", 3)

        # -- Size Breakdown -------------------------------------------- #
        sec2 = self._make_section(self.stats_frame, "Size Breakdown")
        self._add_table(sec2,
                        ["Component", "Bytes", "Bits", "% of Total"],
                        [
                            ["Raw Image (RGB)",
                             f"{s['raw_bytes']:,}",
                             f"{s['raw_bits']:,}",
                             "100.0%"],
                            ["Coefficient Data",
                             f"{(s['coeff_bits'] + 7) // 8:,}",
                             f"{s['coeff_bits']:,}",
                             f"{s['coeff_bits'] / s['raw_bits'] * 100:.1f}%"],
                            ["Header Overhead",
                             f"{(s['header_bits'] + 7) // 8:,}",
                             f"{s['header_bits']:,}",
                             f"{s['header_bits'] / s['raw_bits'] * 100:.1f}%"],
                            ["Total Compressed",
                             f"{s['comp_bytes']:,}",
                             f"{s['comp_bits']:,}",
                             f"{s['comp_bits'] / s['raw_bits'] * 100:.1f}%"],
                        ])

        # -- Per-Channel Breakdown ------------------------------------- #
        sec3 = self._make_section(self.stats_frame, "Per-Channel Breakdown")
        ch_rows = []
        for name in ("Y (luma)", "Co (chroma-orange)", "Cg (chroma-green)"):
            key = name.split()[0]
            ch = s["channels"][key]
            ch_rows.append([
                name,
                f"{ch['bits']:,}",
                f"{ch['bpp']:.3f}",
                f"{ch['ratio']:.3f} : 1",
            ])
        self._add_table(sec3, ["Channel", "Total Bits", "bpp", "Ratio"],
                        ch_rows)

        # -- Lossless Verification ------------------------------------- #
        sec4 = self._make_section(self.stats_frame, "Lossless Verification")
        self._add_metric(sec4, "Max Absolute Error", str(s["max_err"]), 0,
                         val_fg=GREEN if s["max_err"] == 0 else RED)
        self._add_metric(sec4, "Mean Absolute Error",
                         f"{s['mean_err']:.6f}", 1)
        self._add_metric(sec4, "Wrong Pixels",
                         f"{s['wrong_px']:,} / {s['pixels']:,}", 2)
        pct = (s["wrong_px"] / s["pixels"]) * 100.0
        self._add_metric(sec4, "Pixel Error %", f"{pct:.4f}%", 3)

        pass_text = ("PASS — Bit-exact reconstruction" if s["lossless"]
                     else "FAIL — Reconstruction has errors!")
        pass_style = "Pass.TLabel" if s["lossless"] else "Fail.TLabel"
        ttk.Label(self.stats_frame, text=pass_text,
                  style=pass_style).pack(pady=(10, 4))

    def _show_error(self, msg):
        self.status_var.set(f"Error: {msg}")
        self.btn_select.config(state=tk.NORMAL)


if __name__ == "__main__":
    app = DemoApp()
    app.mainloop()
