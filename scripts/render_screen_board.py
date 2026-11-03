import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 13 classes
CLASSES = [
    "empty",
    "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"
]

def draw_digital_piece(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, piece_type: str, color: str):
    """
    Renders high-quality standard 2D digital chess piece artwork onto a PIL Image.
    x, y is the top-left of the square; size is the square size.
    """
    fill_col = (255, 255, 255) if color == 'white' else (45, 45, 45)
    stroke_col = (30, 30, 30) if color == 'white' else (240, 240, 240)
    stroke_w = max(1, int(size * 0.04))

    cx = x + size * 0.5
    cy = y + size * 0.5

    # Base platform for all pieces
    base_w = size * 0.65
    base_h = size * 0.12
    draw.ellipse(
        [cx - base_w * 0.5, y + size * 0.82 - base_h * 0.5, cx + base_w * 0.5, y + size * 0.82 + base_h * 0.5],
        fill=fill_col, outline=stroke_col, width=stroke_w
    )

    if piece_type == 'pawn':
        # Stem
        draw.polygon(
            [(cx - size * 0.16, y + size * 0.8), (cx + size * 0.16, y + size * 0.8),
             (cx + size * 0.1, y + size * 0.5), (cx - size * 0.1, y + size * 0.5)],
            fill=fill_col, outline=stroke_col, width=stroke_w
        )
        # Head ball
        r = size * 0.18
        draw.ellipse([cx - r, y + size * 0.42 - r, cx + r, y + size * 0.42 + r], fill=fill_col, outline=stroke_col, width=stroke_w)

    elif piece_type == 'rook':
        # Body
        draw.polygon(
            [(cx - size * 0.22, y + size * 0.8), (cx + size * 0.22, y + size * 0.8),
             (cx + size * 0.18, y + size * 0.42), (cx - size * 0.18, y + size * 0.42)],
            fill=fill_col, outline=stroke_col, width=stroke_w
        )
        # Castle turrets / battlements
        tw = size * 0.48
        th = size * 0.16
        draw.rectangle([cx - tw*0.5, y + size * 0.28, cx + tw*0.5, y + size * 0.42], fill=fill_col, outline=stroke_col, width=stroke_w)
        # 3 crenellations
        draw.rectangle([cx - tw*0.14, y + size * 0.28, cx + tw*0.14, y + size * 0.35], fill=(fill_col[0]^20, fill_col[1]^20, fill_col[2]^20), outline=stroke_col, width=stroke_w)

    elif piece_type == 'knight':
        # Horse head silhouette
        pts = [
            (cx - size * 0.2, y + size * 0.8),
            (cx + size * 0.2, y + size * 0.8),
            (cx + size * 0.18, y + size * 0.6),
            (cx + size * 0.24, y + size * 0.45),
            (cx + size * 0.12, y + size * 0.26),  # snout tip
            (cx - size * 0.05, y + size * 0.24),  # ear
            (cx - size * 0.22, y + size * 0.4),   # mane
            (cx - size * 0.2, y + size * 0.6),
        ]
        draw.polygon(pts, fill=fill_col, outline=stroke_col, width=stroke_w)
        # Eye dot
        eye_col = stroke_col
        draw.ellipse([cx + size * 0.05, y + size * 0.36, cx + size * 0.09, y + size * 0.4], fill=eye_col)

    elif piece_type == 'bishop':
        # Stem
        draw.polygon(
            [(cx - size * 0.18, y + size * 0.8), (cx + size * 0.18, y + size * 0.8),
             (cx + size * 0.12, y + size * 0.58), (cx - size * 0.12, y + size * 0.58)],
            fill=fill_col, outline=stroke_col, width=stroke_w
        )
        # Mitre oval head
        draw.ellipse([cx - size * 0.18, y + size * 0.28, cx + size * 0.18, y + size * 0.62], fill=fill_col, outline=stroke_col, width=stroke_w)
        # Small cross or ball on top
        draw.ellipse([cx - size * 0.05, y + size * 0.22, cx + size * 0.05, y + size * 0.30], fill=fill_col, outline=stroke_col, width=stroke_w)
        # Slit line
        draw.line([(cx - size * 0.08, y + size * 0.36), (cx + size * 0.06, y + size * 0.48)], fill=stroke_col, width=stroke_w)

    elif piece_type == 'queen':
        # Body
        draw.polygon(
            [(cx - size * 0.22, y + size * 0.8), (cx + size * 0.22, y + size * 0.8),
             (cx + size * 0.14, y + size * 0.5), (cx - size * 0.14, y + size * 0.5)],
            fill=fill_col, outline=stroke_col, width=stroke_w
        )
        # Coronet with 5 spikes
        crown_pts = [
            (cx - size * 0.24, y + size * 0.36),
            (cx - size * 0.14, y + size * 0.46),
            (cx, y + size * 0.32),
            (cx + size * 0.14, y + size * 0.46),
            (cx + size * 0.24, y + size * 0.36),
            (cx + size * 0.16, y + size * 0.52),
            (cx - size * 0.16, y + size * 0.52),
        ]
        draw.polygon(crown_pts, fill=fill_col, outline=stroke_col, width=stroke_w)
        # Little pearls on crown tips
        for px, py in [(cx - size * 0.24, y + size * 0.36), (cx, y + size * 0.32), (cx + size * 0.24, y + size * 0.36)]:
            draw.ellipse([px - size * 0.04, py - size * 0.04, px + size * 0.04, py + size * 0.04], fill=fill_col, outline=stroke_col, width=stroke_w)

    elif piece_type == 'king':
        # Body
        draw.polygon(
            [(cx - size * 0.24, y + size * 0.8), (cx + size * 0.24, y + size * 0.8),
             (cx + size * 0.16, y + size * 0.48), (cx - size * 0.16, y + size * 0.48)],
            fill=fill_col, outline=stroke_col, width=stroke_w
        )
        # Crown dome
        draw.ellipse([cx - size * 0.2, y + size * 0.34, cx + size * 0.2, y + size * 0.52], fill=fill_col, outline=stroke_col, width=stroke_w)
        # Cross on top
        cw = max(2, int(size * 0.06))
        ch = size * 0.16
        draw.rectangle([cx - cw*0.5, y + size * 0.18, cx + cw*0.5, y + size * 0.34], fill=stroke_col)
        draw.rectangle([cx - size * 0.1, y + size * 0.22, cx + size * 0.1, y + size * 0.22 + cw], fill=stroke_col)

print("render_screen_board module ready.")
