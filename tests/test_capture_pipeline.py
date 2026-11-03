#!/usr/bin/env python3
"""Synthetic software tests for the Milestone 5.2H capture pipeline.

These tests use generated 800x800 fixtures only. They are not physical capture
evidence and must never be reported as a completed Set 01 pilot.
"""

import hashlib
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
import benchmark_board_geometry as geom_bench  # noqa: E402

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

    def test_orientation_agreement_accounting_and_symmetry(self):
        # 1. Standard asymmetric board: White King at e1, Black King at e8
        grid_asym = adapter.fen_to_grid("4k3/8/8/8/8/8/8/4K3")
        corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32)
        pieces_asym = [
            {"cx": 0.55, "cy": 0.85, "info": {"char": "K"}},  # e1 (near row 7, col 4)
            {"cx": 0.55, "cy": 0.15, "info": {"char": "k"}},  # e8 (near row 0, col 4)
        ]
        _, ori_name, score, matches, is_tied, best_shifts = adapter.resolve_board_orientation(
            corners, pieces_asym, grid_asym, (1000, 1000)
        )
        self.assertEqual(ori_name, "white_perspective")
        self.assertEqual(matches, 2)
        self.assertEqual(score, 1.0)
        self.assertFalse(is_tied)
        self.assertEqual(best_shifts, [0])  # Unique strict winner

        # 2. 180-degree rotationally symmetric board:
        # e.g., Black King on a8, Black King on h1 (diametrically opposite matching piece symbols)
        grid_sym = adapter.fen_to_grid("k7/8/8/8/8/8/8/7k")
        pieces_sym = [
            {"cx": 0.15, "cy": 0.15, "info": {"char": "k"}},  # a8
            {"cx": 0.85, "cy": 0.85, "info": {"char": "k"}},  # h1
        ]
        _, _, score_sym, matches_sym, is_tied_sym, best_shifts_sym = adapter.resolve_board_orientation(
            corners, pieces_sym, grid_sym, (1000, 1000)
        )
        self.assertTrue(is_tied_sym)
        self.assertIn(0, best_shifts_sym)
        self.assertIn(2, best_shifts_sym)  # Shift 0 and Shift 2 (180 deg) both match!
        self.assertEqual(len(best_shifts_sym), 2)
        self.assertEqual(score_sym, 1.0)

        # 3. Empty board (0 pieces): 4-way tie
        grid_empty = adapter.fen_to_grid("8/8/8/8/8/8/8/8")
        _, _, score_empty, matches_empty, is_tied_empty, best_shifts_empty = adapter.resolve_board_orientation(
            corners, [], grid_empty, (1000, 1000)
        )
        self.assertTrue(is_tied_empty)
        self.assertEqual(best_shifts_empty, [0, 1, 2, 3])
        self.assertEqual(score_empty, 0.0)
        self.assertEqual(matches_empty, 0)

    def test_geometry_benchmark_iou_and_corner_error_metrics(self):
        # Exact match
        sq1 = np.array([[100, 100], [500, 100], [500, 500], [100, 500]], dtype=np.float32)
        sq2 = np.array([[100, 100], [500, 100], [500, 500], [100, 500]], dtype=np.float32)
        iou_exact = geom_bench.compute_polygon_iou(sq1, sq2, (600, 600))
        err_exact = geom_bench.compute_normalized_corner_error(sq1, sq2, (600, 600))
        self.assertAlmostEqual(iou_exact, 1.0, places=4)
        self.assertAlmostEqual(err_exact, 0.0, places=4)

        # Cyclically rolled corners (cyclic permutation invariance)
        sq2_rolled = np.roll(sq2, -1, axis=0)
        err_rolled = geom_bench.compute_normalized_corner_error(sq1, sq2_rolled, (600, 600))
        self.assertAlmostEqual(err_rolled, 0.0, places=4)

        # Non-overlapping
        sq_far = np.array([[600, 600], [800, 600], [800, 800], [600, 800]], dtype=np.float32)
        iou_zero = geom_bench.compute_polygon_iou(sq1, sq_far, (1000, 1000))
        self.assertEqual(iou_zero, 0.0)

        # Shifted by 20 pixels
        sq_shifted = np.array([[120, 120], [520, 120], [520, 520], [120, 520]], dtype=np.float32)
        iou_shifted = geom_bench.compute_polygon_iou(sq1, sq_shifted, (600, 600))
        err_shifted = geom_bench.compute_normalized_corner_error(sq1, sq_shifted, (600, 600))
        self.assertTrue(0.80 <= iou_shifted < 1.0)
        self.assertTrue(0.01 <= err_shifted <= 0.06)

    def test_geometry_benchmark_screen_space_corner_ordering(self):
        # Randomly shuffled corners of a trapezoid
        pts = np.array([[800, 750], [150, 200], [700, 250], [200, 800]], dtype=np.float32)
        ordered = geom_bench.order_corners_screen_space(pts)

        # First must be top-left (min x+y -> [150, 200])
        np.testing.assert_allclose(ordered[0], [150, 200])
        # Second must be top-right -> [700, 250]
        np.testing.assert_allclose(ordered[1], [700, 250])
        # Third must be bottom-right -> [800, 750]
        np.testing.assert_allclose(ordered[2], [800, 750])
        # Fourth must be bottom-left -> [200, 800]
        np.testing.assert_allclose(ordered[3], [200, 800])

    def test_assisted_calibration_simulated_workflow(self):
        # Generate a synthetic 800x800 checkerboard image
        img_path = self.write_synthetic_image("test_assisted_board.png", size=800)
        img = cv2.imread(str(img_path))
        gt_corners = np.array([[100, 100], [700, 100], [700, 700], [100, 700]], dtype=np.float32)

        np.random.seed(42)
        ordered_refined, conf, reason = geom_bench.detect_approach_d_assisted(
            img, gt_corners, noise_std_diag=0.02
        )

        iou = geom_bench.compute_polygon_iou(ordered_refined, gt_corners, (800, 800))
        err = geom_bench.compute_normalized_corner_error(ordered_refined, gt_corners, (800, 800))

        self.assertEqual(reason, "ok")
        self.assertTrue(iou >= 0.85)
        self.assertTrue(err <= 0.04)


class GeometryBenchmarkAccountingTests(unittest.TestCase):
    """Milestone 5.3D: accounting and summary-vs-record reproducibility tests."""

    @staticmethod
    def make_record(
        *,
        approach: str = "test",
        view: str = "frontal",
        candidate_generated: bool = True,
        geometry_valid: bool = True,
        accepted: bool = True,
        rejection_reason=None,
        rejection_stage=None,
        iou: float = 0.8,
        corner_error: float = 0.1,
    ):
        return {
            "approach": approach,
            "file": "synthetic.jpg",
            "view": view,
            "candidate_generated": candidate_generated,
            "geometry_valid": geometry_valid,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "rejection_stage": rejection_stage,
            "candidate_rejection_reason": None,
            "geometry_rejection_reason": None,
            "geometry_rejection_detail": None,
            "iou": iou,
            "corner_error": corner_error,
            "approach_confidence": 0.0,
            "boundary_score": 0.0,
            "combined_confidence": 0.0,
            "pattern": None,
            "sampling_scale": None,
        }

    def test_summary_denominators_and_quality_views_are_derived_from_records(self):
        records = [
            self.make_record(iou=1.0, corner_error=0.0),
            self.make_record(iou=0.6, corner_error=0.2),
            self.make_record(
                accepted=False,
                rejection_reason="pattern_fit_below_threshold",
                rejection_stage="acceptance",
                iou=0.5,
                corner_error=0.3,
            ),
            self.make_record(
                geometry_valid=False,
                accepted=False,
                rejection_reason="geometry_non_convex",
                rejection_stage="geometry",
                iou=0.0,
                corner_error=1.0,
            ),
            self.make_record(
                candidate_generated=False,
                geometry_valid=False,
                accepted=False,
                rejection_reason="no_candidate",
                rejection_stage="candidate",
                iou=0.0,
                corner_error=1.0,
            ),
        ]

        summary = geom_bench.summarize_approach(records)
        accounting = summary["accounting"]

        self.assertEqual(accounting["total_images"], 5)
        self.assertEqual(accounting["candidate_generated_count"], 4)
        self.assertEqual(accounting["geometry_valid_count"], 3)
        self.assertEqual(accounting["accepted_count"], 2)
        self.assertAlmostEqual(accounting["accepted_pct"], 40.0)

        # Accepted view: conditional on the two accepted records.
        self.assertEqual(summary["accepted_quality"]["denominator"], 2)
        self.assertAlmostEqual(summary["accepted_quality"]["mean_iou_pct"], 80.0)
        self.assertAlmostEqual(
            summary["accepted_quality"]["mean_corner_error_pct"], 10.0
        )

        # Proposal view: conditional on the three geometry-valid proposals.
        self.assertEqual(summary["proposal_quality"]["denominator"], 3)
        self.assertAlmostEqual(
            summary["proposal_quality"]["mean_iou_pct"],
            round((1.0 + 0.6 + 0.5) / 3 * 100, 4),
        )

        # All-input view: every image counts; rejected proposals are penalized.
        self.assertEqual(summary["all_input_penalized"]["denominator"], 5)
        self.assertAlmostEqual(
            summary["all_input_penalized"]["mean_iou_pct"],
            round((1.0 + 0.6) / 5 * 100, 4),
        )
        self.assertAlmostEqual(
            summary["all_input_penalized"]["mean_corner_error_pct"],
            round((0.0 + 0.2 + 1.0 + 1.0 + 1.0) / 5 * 100, 4),
        )

        self.assertEqual(
            summary["rejection_reasons"],
            {
                "pattern_fit_below_threshold": 1,
                "geometry_non_convex": 1,
                "no_candidate": 1,
            },
        )
        self.assertEqual(
            summary["rejection_stages"],
            {"acceptance": 1, "geometry": 1, "candidate": 1},
        )

    def test_missing_and_invalid_proposals_receive_documented_penalties(self):
        gt = np.array(
            [[100, 100], [700, 100], [700, 700], [100, 700]], dtype=np.float32
        )
        invalid = {"geometry_valid": False, "points": None}
        iou, error = geom_bench._metric_values(invalid, gt, (800, 800))
        self.assertEqual(iou, 0.0)
        self.assertEqual(error, 1.0)

        valid = {"geometry_valid": True, "points": gt}
        iou, error = geom_bench._metric_values(valid, gt, (800, 800))
        self.assertAlmostEqual(iou, 1.0)
        self.assertAlmostEqual(error, 0.0)

    def test_record_construction_preserves_explicit_accounting_fields(self):
        checkerboard = self.make_checkerboard(800)
        corners = np.array(
            [[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32
        )
        result = geom_bench._finish_candidate(
            corners, (800, 800), 0.35, checkerboard, corners
        )
        accepted, reason, stage = geom_bench.production_gate(result)
        record = geom_bench._make_record(
            "b_relaxed",
            "synthetic_checkerboard.png",
            "frontal",
            result,
            accepted,
            reason,
            stage,
            1.0,
            0.0,
        )

        for field in (
            "candidate_generated",
            "geometry_valid",
            "accepted",
            "rejection_reason",
        ):
            self.assertIn(field, record)
        self.assertTrue(record["candidate_generated"])
        self.assertTrue(record["geometry_valid"])
        self.assertTrue(record["accepted"])
        self.assertIsNone(record["rejection_reason"])
        self.assertTrue(record["pattern"]["valid"])
        self.assertGreaterEqual(record["combined_confidence"], 0.32)

        summary = geom_bench.summarize_approach([record])
        geom_bench.assert_summary_matches_records(summary, [record])
        self.assertEqual(summary["accounting"]["accepted_count"], 1)

    def test_strict_production_gate_requires_pattern_evidence_while_relaxed_gate_does_not(self):
        result = {
            "candidate_generated": True,
            "geometry_valid": True,
            "candidate_rejection_reason": None,
            "geometry_rejection_reason": None,
            "pattern": {
                "valid": False,
                "reason": "pattern_fit_below_threshold",
                "score": 0.0,
            },
            "combined_confidence": 0.0,
        }

        relaxed, relaxed_reason, _ = geom_bench.relaxed_geometry_gate(result)
        strict, strict_reason, strict_stage = geom_bench.production_gate(result)

        self.assertTrue(relaxed)
        self.assertIsNone(relaxed_reason)
        self.assertFalse(strict)
        self.assertEqual(strict_reason, "pattern_fit_below_threshold")
        self.assertEqual(strict_stage, "acceptance")

        result["pattern"] = {"valid": True, "reason": None, "score": 0.9}
        result["combined_confidence"] = 0.8 * 0.9
        strict, strict_reason, strict_stage = geom_bench.production_gate(result)
        self.assertTrue(strict)
        self.assertIsNone(strict_reason)
        self.assertIsNone(strict_stage)

    def test_pattern_evidence_accepts_checkerboard_and_rejects_non_checkerboards(self):
        checkerboard = self.make_checkerboard(800)
        corners = np.array(
            [[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32
        )
        evidence = geom_bench.measure_pattern_evidence(checkerboard, corners)
        self.assertTrue(evidence["valid"])
        self.assertGreaterEqual(evidence["fit"], 0.55)
        self.assertGreaterEqual(evidence["alternation"], 0.80)
        self.assertGreater(evidence["score"], 0.45)

        uniform = np.full((800, 800), 128, dtype=np.uint8)
        uniform_evidence = geom_bench.measure_pattern_evidence(uniform, corners)
        self.assertFalse(uniform_evidence["valid"])
        self.assertEqual(
            uniform_evidence["reason"], "pattern_contrast_below_threshold"
        )

        stripes = np.zeros((800, 800), dtype=np.uint8)
        rows, columns = np.indices((800, 800))
        stripes = (((columns // 100) % 2) * 255).astype(np.uint8)
        stripe_evidence = geom_bench.measure_pattern_evidence(stripes, corners)
        self.assertFalse(stripe_evidence["valid"])
        self.assertIsNotNone(stripe_evidence["reason"])

    def test_geometry_validation_rejects_concave_and_nearly_collinear_quads(self):
        convex = np.array(
            [[100, 100], [700, 100], [700, 700], [100, 700]], dtype=np.float32
        )
        valid, reason, _ = geom_bench.validate_quadrilateral(convex, (800, 800))
        self.assertTrue(valid)
        self.assertIsNone(reason)

        concave = np.array(
            [[50, 50], [750, 50], [250, 250], [50, 750]], dtype=np.float32
        )
        valid, reason, _ = geom_bench.validate_quadrilateral(concave, (800, 800))
        self.assertFalse(valid)
        self.assertEqual(reason, "geometry_non_convex")

        nearly_collinear = np.array(
            [[50, 400], [300, 400], [550, 400.0000001], [400, 750]], dtype=np.float32
        )
        valid, reason, _ = geom_bench.validate_quadrilateral(
            nearly_collinear, (800, 800)
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "geometry_non_convex")

    def test_approach_d_summary_reports_initial_paired_and_all_input_denominators(self):
        def make_d_record(initial_iou, refined_iou, refined_valid):
            return {
                "approach": "d_simulated",
                "file": "synthetic.jpg",
                "view": "frontal",
                "candidate_generated": True,
                "geometry_valid": refined_valid,
                "accepted": refined_valid,
                "rejection_reason": None if refined_valid else "geometry_non_convex",
                "rejection_stage": None if refined_valid else "geometry",
                "iou": refined_iou if refined_valid else 0.0,
                "corner_error": 0.05 if refined_valid else 1.0,
                "initial": {
                    "candidate_generated": True,
                    "geometry_valid": True,
                    "geometry_rejection_reason": None,
                    "rejection_reason": None,
                    "iou": initial_iou,
                    "corner_error": 0.1,
                },
            }

        records = [
            make_d_record(0.80, 0.90, True),
            make_d_record(0.80, 0.90, True),
            make_d_record(0.80, 0.90, True),
            make_d_record(0.80, 0.90, True),
            make_d_record(0.70, 0.0, False),
        ]

        summary = geom_bench.summarize_approach_d(records)

        self.assertEqual(
            summary["initial_baseline_all_input"]["denominator"], 5
        )
        self.assertAlmostEqual(
            summary["initial_baseline_all_input"]["mean_iou_pct"],
            round((0.8 * 4 + 0.7) / 5 * 100, 4),
        )
        self.assertEqual(
            summary["refined_all_input_penalized"]["denominator"], 5
        )
        self.assertEqual(
            summary["refined_all_input_penalized"]["accepted_count"], 4
        )
        self.assertAlmostEqual(
            summary["refined_all_input_penalized"]["mean_iou_pct"],
            round((0.9 * 4) / 5 * 100, 4),
        )

        paired = summary["paired_before_after"]
        self.assertEqual(paired["denominator"], 4)
        self.assertEqual(paired["excluded_count"], 1)
        self.assertAlmostEqual(paired["initial_mean_iou_pct"], 80.0)
        self.assertAlmostEqual(paired["refined_mean_iou_pct"], 90.0)
        self.assertAlmostEqual(paired["mean_delta_iou_pct_points"], 10.0)
        self.assertEqual(paired["improved_iou_count"], 4)
        self.assertEqual(paired["worsened_iou_count"], 0)
        self.assertEqual(paired["unchanged_iou_count"], 0)

        geom_bench.assert_d_summary_matches_records(summary, records)

    def test_summary_consistency_guard_detects_tampered_aggregate(self):
        records = [
            self.make_record(iou=1.0, corner_error=0.0),
            self.make_record(
                accepted=False,
                rejection_reason="no_candidate",
                rejection_stage="candidate",
                iou=0.0,
                corner_error=1.0,
            ),
        ]
        summary = geom_bench.summarize_approach(records)
        geom_bench.assert_summary_matches_records(summary, records)

        summary["accounting"]["accepted_count"] = 2
        with self.assertRaises(AssertionError):
            geom_bench.assert_summary_matches_records(summary, records)

    def test_written_record_hash_matches_exact_file_bytes(self):
        records = [
            self.make_record(iou=0.75, corner_error=0.05),
            self.make_record(
                accepted=False,
                rejection_reason="pattern_fit_below_threshold",
                rejection_stage="acceptance",
                iou=0.25,
                corner_error=0.4,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "geometry_records.jsonl"
            metadata = geom_bench.write_records(path, records)
            payload = path.read_bytes()

            self.assertEqual(metadata["records_count"], 2)
            self.assertEqual(
                metadata["records_sha256"],
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertNotIn(b"\r\n", payload)
            loaded = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
            self.assertEqual(len(loaded), 2)

    @staticmethod
    def make_checkerboard(size: int):
        rows, columns = np.indices((size, size))
        square = size // 8
        return ((((rows // square) + (columns // square)) % 2) * 255).astype(
            np.uint8
        )


if __name__ == "__main__":
    unittest.main()
