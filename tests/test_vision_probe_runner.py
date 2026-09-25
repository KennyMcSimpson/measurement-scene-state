import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_vision_probe.py"
_SPEC = importlib.util.spec_from_file_location("run_vision_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["run_vision_probe"] = _MODULE
_SPEC.loader.exec_module(_MODULE)
mask_metrics = _MODULE.mask_metrics
run = _MODULE.run


class FakeExtractor:
    calls: list[Path] = []

    def extract(self, path: Path):
        self.calls.append(path)
        if "sealed" in path.parts:
            raise AssertionError("sealed sequence image was read")
        seed = int(hashlib.sha256(path.read_bytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(4, 4, 8)).astype("float32")
        rgb = rng.random((4, 4, 3), dtype=np.float32)
        return features, rgb, (16, 16), 224


def _make_dataset(root: Path) -> None:
    dataset = root / "DAVIS"
    image_root = dataset / "JPEGImages" / "480p"
    mask_root = dataset / "Annotations" / "480p"
    split_root = dataset / "ImageSets" / "2017"
    names = ["fit_a", "fit_b", "discovery", "validation", "sealed"]
    for split, split_names in {"train": names[:4], "val": names[4:]}.items():
        (split_root / f"{split}.txt").parent.mkdir(parents=True, exist_ok=True)
        (split_root / f"{split}.txt").write_text("\n".join(split_names) + "\n", encoding="utf-8")
    for name in names:
        image_dir = image_root / name
        mask_dir = mask_root / name
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for index in range(3):
            (image_dir / f"{index:05d}.jpg").write_bytes(f"{name}-{index}".encode())
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[:, index : index + 2] = 1
            Image.fromarray(mask).save(mask_dir / f"{index:05d}.png")


def test_runner_keeps_official_val_sealed_and_writes_reports(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fixture weights")
    extractor = FakeExtractor()
    args = argparse.Namespace(
        data_root=tmp_path,
        weights=weights,
        output=tmp_path / "output",
        device="cpu",
        size=224,
        fit_sequences=1,
        discovery_sequences=1,
        validation_sequences=1,
        eta=[0.2],
    )
    summary = run(args, extractor=extractor)
    assert summary["status"] == "complete"
    assert summary["split_counts"] == {"fit": 1, "discovery": 1, "validation": 1, "sealed": 2}
    assert all("sealed" not in path.parts for path in extractor.calls)
    assert (args.output / "readout.pt").is_file()
    raw_rows = [
        json.loads(line)
        for line in (args.output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation_rows = [row for row in raw_rows if row["split"] == "validation"]
    expected_methods = {
        "OFFOFF",
        "AA",
        "BB",
        "ALLALL",
        "residual_selector",
        "bestgloballyfixed",
    }
    target_keys_by_method = {
        method: {
            (row["sequence"], row["target_slot"])
            for row in validation_rows
            if row["method"] == method
        }
        for method in expected_methods
    }
    assert len(target_keys_by_method["OFFOFF"]) == 2
    assert all(keys == target_keys_by_method["OFFOFF"] for keys in target_keys_by_method.values())
    assert '"status": "complete"' in (args.output / "status.json").read_text(encoding="utf-8")


def test_mask_metrics_uses_nonzero_object_union(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    source_mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    target_mask = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    Image.fromarray(source_mask).save(source)
    Image.fromarray(target_mask).save(target)
    features = np.eye(4, dtype=np.float32).reshape(2, 2, 4)
    result = mask_metrics(
        torch.from_numpy(features),
        torch.from_numpy(features),
        source,
        target,
    )
    assert result["n_objects"] == 2
    assert result["mean_iou"] == 0.25
