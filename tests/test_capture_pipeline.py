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
import external_benchmark_adapter as adapter  # noqa: E402

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

    def test_external_adapter_parses_placement_formats(self):
        # 1. Underscore separator with side marker
        p1 = adapter.parse_fenit_placement("1k1r3r_7p_p1p1Rp2_1pbp3b_5P1P_6PB_PPP5_RK6 w")
        self.assertEqual(p1, "1k1r3r/7p/p1p1Rp2/1pbp3b/5P1P/6PB/PPP5/RK6")

        # 2. Hyphen separator
        p2 = adapter.parse_fenit_placement("2b5-4kp2-p6p-2Np1P2-PP1Nn1p1-6P1-7P-4K3")
        self.assertEqual(p2, "2b5/4kp2/p6p/2Np1P2/PP1Nn1p1/6P1/7P/4K3")

        # 3. Trailing '(full board)'
        p3 = adapter.parse_fenit_placement("rnbqkbnr_pppppppp_8_8_8_8_PPPPPPPP_RNBQKBNR (full board)")
        self.assertEqual(p3, START_PLACEMENT)

        # 4. Ineligible: timestamp stem
        self.assertIsNone(adapter.parse_fenit_placement("20260219_220620"))

        # 5. Ineligible: single piece stem
        self.assertIsNone(adapter.parse_fenit_placement("B"))
        self.assertIsNone(adapter.parse_fenit_placement("K"))

    def test_external_adapter_detects_non_convex_corners(self):
        # Strictly convex quadrilateral
        convex = np.array([[0.1, 0.1], [0.9, 0.1], [0.8, 0.8], [0.15, 0.85]], dtype=np.float32)
        self.assertTrue(adapter.is_strictly_convex(convex))

        # Bowtie / self-intersecting quadrilateral
        bowtie = np.array([[0.1, 0.1], [0.8, 0.8], [0.9, 0.1], [0.15, 0.85]], dtype=np.float32)
        self.assertFalse(adapter.is_strictly_convex(bowtie))

    def test_external_adapter_emits_observation_records_without_physical_identities(self):
        # Create a mock 1000x1000 board image
        img_path = self.write_synthetic_image("fenit_mock_board.png", size=1000)
        ann_path = self.raw_dir / "fenit_mock_board.txt"

        # Annotation: Board line with 4 corners (0.1, 0.1) -> (0.9, 0.1) -> (0.9, 0.9) -> (0.1, 0.9)
        # and two piece lines: White King at center e1, Black King at e8
        ann_lines = [
            "0 0.5 0.5 0.8 0.8 0.1 0.1 2 0.9 0.1 2 0.9 0.9 2 0.1 0.9 2",
            "1 0.55 0.85 0.05 0.08 0.55 0.85 2 0.55 0.85 2 0.55 0.85 2 0.55 0.85 2",
            "4 0.55 0.15 0.05 0.08 0.55 0.15 2 0.55 0.15 2 0.55 0.15 2 0.55 0.15 2",
        ]
        ann_path.write_text("\n".join(ann_lines), encoding="utf-8")

        # Mock image name encoding FEN placement
        mock_placement_stem = "4k3_8_8_8_8_8_8_4K3"
        mock_img_path = self.raw_dir / f"{mock_placement_stem}.png"
        os.rename(img_path, mock_img_path)
        mock_ann_path = self.raw_dir / f"{mock_placement_stem}.txt"
        os.rename(ann_path, mock_ann_path)

        out_dir = self.root / "external_benchmark_out"
        rec = adapter.process_external_image(
            mock_img_path, mock_ann_path, out_dir, W_model=None, b_model=None, temp_model=1.0
        )

        self.assertEqual(rec["status"], "PROCESSED")
        self.assertEqual(rec["dataset"], "FENiT_v1")
        self.assertEqual(rec["derived_crops_count"], 64)
        self.assertTrue(os.path.isfile(rec["rectified_board_path"]))

        # Invariant 1: No SETxx physical piece IDs or physical tag IDs in external records
        self.assertNotIn("piece_instance_id", rec)
        self.assertNotIn("physical_tag_id", rec)
        self.assertNotIn("set_id", rec)

        # Invariant 2: Crops have crop_type "external_observation" and no physical tags
        for crop in rec["crops"]:
            self.assertEqual(crop["crop_type"], "external_observation")
            self.assertNotIn("piece_instance_id", crop)
            self.assertNotIn("physical_tag_id", crop)
            self.assertTrue(os.path.isfile(crop["crop_path"]))

        # Invariant 3: Evaluation modes distinguish Mode 1 (Oracle) and Mode 2 (Image-driven)
        modes = rec["evaluation_modes"]
        self.assertTrue(modes["mode1_ground_truth_assisted"]["is_oracle_assisted"])
        self.assertFalse(modes["mode2_image_driven"]["is_oracle_assisted"])
        self.assertEqual(modes["mode2_image_driven"]["resolved_orientation"], "canonical_screen_space")

    def test_external_adapter_distinguishes_oracle_from_image_driven_modes(self):
        # Test image-driven corner ordering produces top-left first in screen space
        corners = np.array([[0.8, 0.8], [0.1, 0.1], [0.85, 0.15], [0.15, 0.85]], dtype=np.float32)
        ordered = adapter.resolve_image_driven_orientation(corners, (1000, 1000))
        # First point must be top-left (min x+y -> [100, 100])
        np.testing.assert_allclose(ordered[0], [100.0, 100.0], atol=1.0)
        # Second point must be top-right ([850, 150])
        np.testing.assert_allclose(ordered[1], [850.0, 150.0], atol=1.0)
        # Third point must be bottom-right ([800, 800])
        np.testing.assert_allclose(ordered[2], [800.0, 800.0], atol=1.0)
        # Fourth point must be bottom-left ([150, 850])
        np.testing.assert_allclose(ordered[3], [150.0, 850.0], atol=1.0)

    def test_selective_accuracy_accounting_uses_independent_numerators(self):
        # Milestone 5.2K: Fixture where:
        # - Sample 1: accepted and correct
        # - Sample 2: accepted and incorrect
        # - Sample 3: abstained and correct (abstained correct prediction)
        # - Sample 4: abstained and incorrect
        samples = [
            {"confidence": 0.85, "margin": 0.35, "is_correct": True},   # accepted, correct
            {"confidence": 0.70, "margin": 0.25, "is_correct": False},  # accepted, incorrect
            {"confidence": 0.40, "margin": 0.10, "is_correct": True},   # abstained, correct
            {"confidence": 0.30, "margin": 0.05, "is_correct": False},  # abstained, incorrect
        ]

        res = adapter.compute_selective_accuracy_accounting(samples)

        # 1. Total samples and partition coverage
        self.assertEqual(res["total_samples"], 4)
        self.assertEqual(res["accepted_count"], 2)
        self.assertEqual(res["abstention_count"], 2)
        self.assertEqual(res["coverage"], 0.50)
        self.assertEqual(res["abstention_rate"], 0.50)

        # 2. Independent numerators check:
        # Overall correct numerator is 2 (sample 1 + sample 3)
        self.assertEqual(res["overall_correct"], 2)
        self.assertEqual(res["overall_accuracy"], 0.50)

        # Accepted & correct numerator is 1 (only sample 1)
        self.assertEqual(res["accepted_and_correct"], 1)
        self.assertEqual(res["accepted_and_incorrect"], 1)

        # Abstained & correct is 1 (sample 3, diagnostic only)
        self.assertEqual(res["abstained_and_correct"], 1)
        self.assertEqual(res["abstained_and_incorrect"], 1)

        # 3. Selective accuracy must be accepted_and_correct / accepted = 1 / 2 = 0.50
        self.assertEqual(res["selective_accuracy"], 0.50)
        self.assertEqual(res["error_rate_among_accepted"], 0.50)

        # 4. Proves the flawed formula (overall_correct / accepted = 2 / 2 = 1.0) is wrong and distinct
        flawed_acc = res["overall_correct"] / res["accepted_count"]
        self.assertNotEqual(res["selective_accuracy"], flawed_acc)
        self.assertEqual(flawed_acc, 1.0)
        self.assertEqual(res["selective_accuracy"], 0.50)

    def test_external_adapter_skips_timestamp_images_cleanly(self):
        img_path = self.write_synthetic_image("20260219_220620.jpg", size=1000)
        ann_path = self.raw_dir / "20260219_220620.txt"
        ann_path.write_text("0 0.5 0.5 0.8 0.8 0.1 0.1 2 0.9 0.1 2 0.9 0.9 2 0.1 0.9 2\n", encoding="utf-8")

        out_dir = self.root / "external_benchmark_out"
        rec = adapter.process_external_image(
            img_path, ann_path, out_dir, W_model=None, b_model=None, temp_model=1.0
        )

        self.assertEqual(rec["status"], "SKIPPED")
        self.assertEqual(rec["reason"], "timestamp_named_no_ground_truth_fen")



if __name__ == "__main__":
    unittest.main()
