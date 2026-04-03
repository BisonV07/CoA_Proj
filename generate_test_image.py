"""Generate a test image with smooth gradients and some detail for compression testing."""

import numpy as np
from PIL import Image


def generate_test_image(width=256, height=256, output_path="test_images/test_gradient.png"):
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for r in range(height):
        for c in range(width):
            img[r, c, 0] = int(128 + 80 * np.sin(2 * np.pi * r / height))
            img[r, c, 1] = int(128 + 80 * np.sin(2 * np.pi * c / width))
            img[r, c, 2] = int(128 + 60 * np.sin(2 * np.pi * (r + c) / (width + height)))

    Image.fromarray(img).save(output_path)
    print(f"Saved test image: {output_path} ({width}x{height})")
    return output_path


if __name__ == "__main__":
    generate_test_image()
