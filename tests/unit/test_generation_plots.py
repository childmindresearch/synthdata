"""Symlink safety tests for generation plot output paths."""

from pathlib import Path

import pytest

from synthdata.evaluation.syntheval_eval import _safe_output_directory


def test_generation_output_parent_symlink_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "generation"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _safe_output_directory(tmp_path, "generation/model", "Generation plot output")


def test_generation_output_final_symlink_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "generation"
    output.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (output / "model.png").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        target = output / "model.png"
        if target.is_symlink():
            raise ValueError("Generation plot output must not be a symlink")
