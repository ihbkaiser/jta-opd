from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from b200_experiment.trainer import _save_checkpoint


class _FakeDistributed:
    """A no-process-group context for checkpoint filesystem tests."""

    def __init__(self, rank: int, world_size: int = 1):
        self.rank = rank
        self.local_rank = rank
        self.world_size = world_size
        self.device = None
        self.barrier_calls = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        self.barrier_calls += 1

    def broadcast_object(self, value, source: int = 0):
        del source
        # These tests exercise the filesystem guard; a real process group
        # supplies the same rank-0 payload to every worker.
        return value if self.is_main else None


class CheckpointSavingTests(unittest.TestCase):
    def test_rank0_removes_stale_staging_directory_before_retry(self):
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            temporary = output_dir / ".checkpoint-000007.incomplete"
            temporary.mkdir()
            (temporary / "stale.marker").write_text("stale", encoding="utf-8")

            with patch("b200_experiment.trainer._save_inference_snapshot"):
                checkpoint = _save_checkpoint(
                    None,
                    None,
                    None,
                    output_dir,
                    step=7,
                    final=False,
                    save_optimizer=False,
                    distributed=_FakeDistributed(rank=0),
                )

            self.assertEqual(checkpoint, output_dir / "checkpoint-000007")
            self.assertTrue(checkpoint.is_dir())
            self.assertFalse(temporary.exists())
            self.assertEqual(
                (output_dir / "latest.json").read_text(encoding="utf-8").strip(),
                '{"step": 7, "checkpoint": "checkpoint-000007", "final": false}',
            )

    def test_rank0_never_overwrites_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            checkpoint = output_dir / "checkpoint-000008"
            checkpoint.mkdir()
            marker = checkpoint / "keep.marker"
            marker.write_text("keep", encoding="utf-8")

            with patch(
                "b200_experiment.trainer._save_inference_snapshot"
            ) as snapshot:
                with self.assertRaises(FileExistsError):
                    _save_checkpoint(
                        None,
                        None,
                        None,
                        output_dir,
                        step=8,
                        final=False,
                        save_optimizer=False,
                        distributed=_FakeDistributed(rank=0),
                    )

            self.assertTrue(marker.exists())
            snapshot.assert_not_called()

    def test_non_main_rank_does_not_reject_rank0_staging_directory(self):
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            temporary = output_dir / ".checkpoint-000009.incomplete"
            temporary.mkdir()

            # Rank 0 has already created this directory, while a slower rank
            # enters _save_checkpoint.  The worker must not inspect the path or
            # report it as a collision.
            with patch(
                "b200_experiment.trainer._save_inference_snapshot"
            ) as snapshot, patch.object(
                Path, "exists", side_effect=AssertionError("non-main path check")
            ), patch.object(
                Path,
                "is_symlink",
                side_effect=AssertionError("non-main symlink check"),
            ):
                checkpoint = _save_checkpoint(
                    None,
                    None,
                    None,
                    output_dir,
                    step=9,
                    final=False,
                    save_optimizer=False,
                    distributed=_FakeDistributed(rank=1, world_size=2),
                )

            self.assertEqual(checkpoint, output_dir / "checkpoint-000009")
            snapshot.assert_called_once()
            self.assertFalse((output_dir / "checkpoint-000009").exists())

    def test_non_main_rank_does_not_reject_existing_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            (output_dir / "checkpoint-000010").mkdir()

            # This is mostly a guard against accidentally reintroducing an
            # all-rank preflight check.  In normal execution the rank has
            # already crossed the setup barrier before rank 0 commits.
            with patch("b200_experiment.trainer._save_inference_snapshot"):
                checkpoint = _save_checkpoint(
                    None,
                    None,
                    None,
                    output_dir,
                    step=10,
                    final=False,
                    save_optimizer=False,
                    distributed=_FakeDistributed(rank=1, world_size=2),
                )

            self.assertEqual(checkpoint, output_dir / "checkpoint-000010")


if __name__ == "__main__":
    unittest.main()
