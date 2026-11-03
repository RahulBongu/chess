#!/usr/bin/env python3
"""
generate_digital_crops.py - Generates clean-room digital piece crops for training.

Renders 200 crops per class (13 classes = 2600 crops total) of size 72x72 across
multiple board themes (green, brown, blue, gray, monochrome) with scale, position,
and background color variations.
"""

from pathlib import Path
import json
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image, ImageDraw

from scripts.render_screen_board import draw_digital_piece, CLASSES
from scripts.train_rgb_square_model import LOGICAL_SIZE

THEMES = [
    # Light, Dark, Name
    ((238, 238, 210), (118, 150, 86), "green"),
    ((240, 217, 181), (181, 136, 99), "brown"),
    ((222, 227, 230), (140, 162, 173), "blue"),
    ((220, 220, 220), (130, 130, 130), "gray"),
    ((245, 245, 245), (160, 160, 160), "mono"),
]

def generate_crops(output_dir: Path, crops_per_class: int = 200, seed: int = 20261104):
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    size = LOGICAL_SIZE  # 72

    manifest = []
    print(f"Generating {crops_per_class} digital crops per class into {output_dir}...")

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(crops_per_class):
            light_col, dark_col, theme_name = rng.choice(THEMES)
            is_light_square = rng.choice([True, False])
            base_color = light_col if is_light_square else dark_col

            # Slight color perturbation for ambient lighting
            r = min(255, max(0, int(base_color[0] + rng.randint(-12, 12))))
            g = min(255, max(0, int(base_color[1] + rng.randint(-12, 12))))
            b = min(255, max(0, int(base_color[2] + rng.randint(-12, 12))))
            bg_color = (r, g, b)

            img = Image.new("RGB", (size, size), bg_color)
            draw = ImageDraw.Draw(img)

            if class_name != "empty":
                # Parse color and type
                piece_color, piece_type = class_name.split("_")
                # Scale piece between 0.80 and 1.05 of square
                piece_scale = rng.uniform(0.82, 1.02)
                p_size = size * piece_scale
                # Center with slight jitter
                jx = rng.uniform(-3.0, 3.0)
                jy = rng.uniform(-3.0, 3.0)
                px = (size - p_size) * 0.5 + jx
                py = (size - p_size) * 0.5 + jy
                draw_digital_piece(draw, px, py, p_size, piece_type, piece_color)

            # Optional subtle scanlines (50% of crops)
            if rng.random() < 0.5:
                arr = np.array(img).astype(np.float32)
                scan_strength = rng.uniform(0.92, 0.98)
                arr[::2] *= scan_strength
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

            crop_path = class_dir / f"crop_{i:04d}_{theme_name}.png"
            img.save(crop_path)
            manifest.append({
                "path": str(crop_path),
                "label": class_idx,
                "class_name": class_name,
                "group": f"digital_{theme_name}",
                "source": "digital:train",
            })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {len(manifest)} digital crops. Manifest: {manifest_path}")
    return manifest

if __name__ == "__main__":
    generate_crops(Path(r"C:\dev\chess\dist\cache\digital_crops"))
