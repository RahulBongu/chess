#!/usr/bin/env python3
"""Synthetic software tests for the Milestone 5.2H capture pipeline.

These tests use generated 800x800 fixtures only. They are not physical capture
evidence and must never be reported as a completed Set 01 pilot.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import extract_square_crops as extractor  # noqa: E402
import validate_capture_manifest as validator  # noqa: E402

START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
START_FEN = f"{START_PLACEMENT} w KQkq - 0 1"

START_PIECE_MAP = {
    "a8": "SET01_B_R01",
    "b8": "SET01_B_N01",
    "c8": "SET01_B_B01",
    "d8": "SET01_B_Q01",
    "e8": "SET01_B_K01",
    "f8": "SET01_B_B02",
    "g8": "SET01_B_N02",
    "h8": "SET01_B_R02",
    "a7": "SET01_B_P01",
    "b7": "SET01_B_P02",
    "c7": "SET01_B_P03",
    "d7": "SET01_B_P04",
    "e7": "SET01_B_P05",
    "f7": "SET01_B_P06",
    "g7": "SET01_B_P07",
    "h7": "SET01_B_P08",
    "a2": "SET01_W_P01",
    "b2": "SET01_W_P02",
    "c2": "SET01_W_P03",
    "d2": "SET01_W_P04",
    "e2": "SET01_W_P05",
    "f2": "SET01_W_P06",
    "g2": "SET01_W_P07",
    "h2": "SET01_W_P08",
    "a1": "SET01_W_R01",
    "b1": "SET01_W_N01",
    "c1": "SET01_W_B01",
    "d1": "SET01_W_Q01",
    "e1": "SET01_W_K01",
    "f1": "SET01_W_B02",
    "g1": "SET01_W_N02",
    "h1": "SET01_W_R02",
}

ALL_PIECE_IDS = [
    *[f"SET01_W_P{i:02d}" for i in range(1, 9)],
    *[f"SET01_B_P{i:02d}" for i in range(1, 9)],
    "SET01_W_N01",
    "SET01_W_N02",
    "SET01_B_N01",
    "SET01_B_N02",
    "SET01_W_B01",
    "SET01_W_B02",
    "SET01_B_B01",
    "SET01_B_B02",
    "SET01_W_R01",
    "SET01_W_R02",
    "SET01_B_R01",
    "SET01_B_R02",
    "SET01_W_Q01",
    "SET01_W_K01",
    "SET01_B_Q01",
    "SET01_B_K01",
]


def single_piece_placement(piece_char: str, square: str) -> str:
    file_index = ord(square[0]) - 97
    rank = int(square[1])
    ranks = []
    for current_rank in range(8, 0, -1):
        if current_rank != rank:
            ranks.append("8")
        else:
            before = "" if file_index == 0 else str(file_index)
            after = "" if file_index == 7 else str(7 - file_index)
            ranks.append(f"{before}{piece_char}{after}")
    return "/".join(ranks)


def piece_char_from_canonical_id(piece_instance_id: str) -> str:
    _, color, rest = piece_instance_id.split("_")
    char = rest[0]
    return char if color == "W" else char.lower()


class CapturePipelineSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="chess-capture-test-")
        self.root = Path(self.temp_dir.name)
        self.raw_dir = self.root / "raw"
        self.crop_dir = self.root / "crops"
        self.manifest_dir = self.root / "manifests"
        self.raw_dir.mkdir()
        self.crop_dir.mkdir()
        self.manifest_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_synthetic_image(self, name: str = "board.png", size: int = 800) -> Path:
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, :] = (40, 80, 120)
        for row in range(0, size, 100):
            for col in range(0, size, 100):
                if (row // 100 + col // 100) % 2 == 0:
                    image[row : row + 100, col : col + 100] = (210, 210, 210)
        path = self.raw_dir / name
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def write_records(self, plan, image_path, scene_id, board_fen=None, manifest_mode="append"):
        return extractor.write_crops_and_manifest(
            plan=plan,
            image_path=str(image_path),
            out_dir=str(self.crop_dir / "train" / "SET01"),
            scene_id=scene_id,
            set_id="SET01",
            split="train",
            board_fen=board_fen,
            camera_angle="zenith",
            lighting_setup="diffuse_daylight",
            photographer="Test Operator",
            rights_holder="Test Operator",
            release_record="synthetic_test_release.pdf",
            occlusion_level="none",
            annotation_status="verified",
            quality_notes="synthetic software test fixture; not physical capture evidence",
            manifest_out=str(self.manifest_dir / "SET01_manifest.jsonl"),
            manifest_mode=manifest_mode,
            overwrite=False,
        )

    def test_full_board_emits_64_and_validates_against_files(self):
        image_path = self.write_synthetic_image()
        image = cv2.imread(str(image_path))
        plan = extractor.plan_crop_extraction(
            image,
            mode="full_board_context",
            board_fen=START_FEN,
            set_id="SET01",
            piece_instance_map=START_PIECE_MAP,
        )

        self.assertEqual(plan["source_photo_count"], 1)
        self.assertEqual(plan["derived_crop_count"], 64)
        records = self.write_records(plan, image_path, "SET01_SCENE_START_ZENITH", START_FEN)

        self.assertEqual(len(records), 64)
        self.assertEqual(len({record["derived_crop_path"] for record in records}), 64)
        self.assertEqual(len({os.path.basename(record["derived_crop_path"]) for record in records}), 64)
        self.assertEqual({record["original_image_path"] for record in records}, {str(image_path.resolve())})
        self.assertTrue(all(os.path.isfile(record["derived_crop_path"]) for record in records))

        valid, errors, summary = validator.validate_manifest(records, check_files=True)
        self.assertTrue(valid, "; ".join(errors))
        self.assertEqual(summary["source_photographs"], 1)
        self.assertEqual(summary["derived_crops"], 64)
        self.assertEqual(summary["crops_by_type"]["full_board_context"], 64)

    def test_occupied_target_only_emits_one(self):
        image_path = self.write_synthetic_image("sweep_pawn_e4.png")
        image = cv2.imread(str(image_path))
        placement = single_piece_placement("P", "e4")
        plan = extractor.plan_crop_extraction(
            image,
            mode="occupied_target_only",
            placement=placement,
            set_id="SET01",
            target_square="e4",
            target_piece_instance_id="SET01_W_P01",
        )

        self.assertEqual(plan["source_photo_count"], 1)
        self.assertEqual(plan["derived_crop_count"], 1)
        self.assertEqual(len(plan["crops"]), 1)
        self.assertEqual(plan["crops"][0]["square"], "e4")
        self.assertEqual(plan["crops"][0]["piece_instance_id"], "SET01_W_P01")

        records = self.write_records(plan, image_path, "SET01_SWEEP_W_P01_E4")
        self.assertEqual(len(records), 1)
        valid, errors, summary = validator.validate_manifest(records, check_files=True)
        self.assertTrue(valid, "; ".join(errors))
        self.assertEqual(summary["source_photographs"], 1)
        self.assertEqual(summary["derived_crops"], 1)

    def test_32_reference_photos_emit_32_records(self):
        records = []
        for piece_id in ALL_PIECE_IDS:
            image_path = self.write_synthetic_image(f"ref_{piece_id}.png")
            image = cv2.imread(str(image_path))
            char = piece_char_from_canonical_id(piece_id)
            placement = single_piece_placement(char, "d4")
            plan = extractor.plan_crop_extraction(
                image,
                mode="occupied_target_only",
                placement=placement,
                set_id="SET01",
                target_square="d4",
                target_piece_instance_id=piece_id,
            )
            self.assertEqual(plan["derived_crop_count"], 1)
            records.extend(
                self.write_records(plan, image_path, f"SET01_REF_{piece_id}")
            )

        self.assertEqual(len(ALL_PIECE_IDS), 32)
        self.assertEqual(len(records), 32)
        valid, errors, summary = validator.validate_manifest(records, check_files=True)
        self.assertTrue(valid, "; ".join(errors))
        self.assertEqual(summary["source_photographs"], 32)
        self.assertEqual(summary["derived_crops"], 32)
        self.assertEqual(len(summary["distinct_piece_instances"]), 32)

    def test_empty_board_grid_emits_64_empty_crops(self):
        image_path = self.write_synthetic_image("empty.png")
        image = cv2.imread(str(image_path))
        plan = extractor.plan_crop_extraction(
            image,
            mode="empty_board_grid",
            set_id="SET01",
        )

        self.assertEqual(plan["derived_crop_count"], 64)
        self.assertTrue(all(crop["piece_instance_id"] == "EMPTY" for crop in plan["crops"]))
        self.assertTrue(all(crop["physical_tag_id"] == "NONE" for crop in plan["crops"]))

        records = self.write_records(plan, image_path, "SET01_EMPTY_DIFFUSE")
        valid, errors, summary = validator.validate_manifest(records, check_files=True)
        self.assertTrue(valid, "; ".join(errors))
        self.assertEqual(summary["source_photographs"], 1)
        self.assertEqual(summary["derived_crops"], 64)
        self.assertEqual(summary["crops_by_type"]["empty_square"], 64)

    def test_invalid_target_identity_and_ambiguous_occupancy_are_rejected(self):
        image = cv2.imread(str(self.write_synthetic_image("invalid_target.png")))
        one_pawn = single_piece_placement("P", "e4")

        with self.assertRaisesRegex(extractor.ExtractionError, "does not match the only occupied square"):
            extractor.plan_crop_extraction(
                image,
                mode="occupied_target_only",
                placement=one_pawn,
                set_id="SET01",
                target_square="d4",
                target_piece_instance_id="SET01_W_P01",
            )

        with self.assertRaisesRegex(extractor.ExtractionError, "is a queen, expected pawn"):
            extractor.plan_crop_extraction(
                image,
                mode="occupied_target_only",
                placement=one_pawn,
                set_id="SET01",
                target_square="e4",
                target_piece_instance_id="SET01_W_Q01",
            )

        with self.assertRaisesRegex(extractor.ExtractionError, "exactly one occupied square"):
            extractor.plan_crop_extraction(
                image,
                mode="occupied_target_only",
                placement=START_PLACEMENT,
                set_id="SET01",
                target_square="e4",
                target_piece_instance_id="SET01_W_P01",
            )

        with self.assertRaisesRegex(extractor.ExtractionError, "requires --piece-map"):
            extractor.plan_crop_extraction(
                image,
                mode="full_board_context",
                placement=START_PLACEMENT,
                set_id="SET01",
            )

    def test_non_800_image_is_rejected_without_corners(self):
        image_path = self.write_synthetic_image("oblique.png", size=600)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "extract_square_crops.py"),
                "--image",
                str(image_path),
                "--mode",
                "empty_board_grid",
                "--out-dir",
                str(self.crop_dir / "train" / "SET01"),
                "--scene-id",
                "SET01_TEST_OBLIQUE",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("will not silently resize", result.stderr)
        self.assertFalse(any((self.crop_dir / "train" / "SET01").glob("*.png")))

    def test_manifest_check_files_detects_missing_crop(self):
        image_path = self.write_synthetic_image("missing_crop.png")
        image = cv2.imread(str(image_path))
        plan = extractor.plan_crop_extraction(
            image,
            mode="occupied_target_only",
            placement=single_piece_placement("P", "e4"),
            set_id="SET01",
            target_square="e4",
            target_piece_instance_id="SET01_W_P01",
        )
        records = self.write_records(plan, image_path, "SET01_SWEEP_MISSING")
        valid, _, _ = validator.validate_manifest(records, check_files=True)
        self.assertTrue(valid)
        os.remove(records[0]["derived_crop_path"])
        valid, errors, _ = validator.validate_manifest(records, check_files=True)
        self.assertFalse(valid)
        self.assertTrue(any("[Missing File]" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
