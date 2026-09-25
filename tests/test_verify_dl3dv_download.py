import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcss.data.dl3dv_download import Inventory, RemoteFile, SceneMeta

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_dl3dv_download.py"
_SPEC = importlib.util.spec_from_file_location("verify_dl3dv_download", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
verify_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_script)


def _remote(path: str, payload: bytes, *, lfs: bool = False) -> RemoteFile:
    size = len(payload)
    git_blob = hashlib.sha1(f"blob {size}\0".encode() + payload).hexdigest()
    return RemoteFile(
        path=path,
        size=size,
        git_blob_id=git_blob,
        lfs_sha256=hashlib.sha256(payload).hexdigest() if lfs else None,
        lfs_size=size if lfs else None,
        scene_hash=path.split("/", 1)[0],
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame_names: tuple[str, ...] = ("frame_00001.png",),
    inventory_extra_names: tuple[str, ...] = (),
    split_indices: tuple[int, ...] = (0,),
    physical_images: tuple[str, ...] | None = None,
):
    scene = "a" * 64
    metadata = f"hash,scene\n{scene},Synthetic\n".encode()
    transform_payload = json.dumps(
        {"frames": [{"file_path": f"images/{name}"} for name in frame_names]}
    ).encode()
    transform = _remote(f"{scene}/nerfstudio/transforms.json", transform_payload)
    images = tuple(
        _remote(
            f"{scene}/nerfstudio/images_4/{name}",
            f"payload-{index}".encode(),
            lfs=True,
        )
        for index, name in enumerate(frame_names + inventory_extra_names)
    )
    metadata_remote = RemoteFile(
        "benchmark-meta.csv",
        len(metadata),
        hashlib.sha1(f"blob {len(metadata)}\0".encode() + metadata).hexdigest(),
        None,
        None,
        None,
    )
    inventory = Inventory(
        metadata,
        metadata_remote,
        (SceneMeta(0, scene, {"scene": "Synthetic"}),),
        (transform, *images),
    )
    output_dir = tmp_path / "inventory"
    data_root = tmp_path / "data"
    output_dir.mkdir()
    transform_path = data_root / transform.path
    transform_path.parent.mkdir(parents=True)
    transform_path.write_bytes(transform_payload)
    if physical_images is None:
        physical_images = frame_names
    all_image_names = frame_names + inventory_extra_names
    image_by_name = {name: image for name, image in zip(all_image_names, images, strict=True)}
    for name in physical_images:
        image_path = data_root / image_by_name[name].path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"payload-{all_image_names.index(name)}".encode())
    (output_dir / "benchmark-meta.csv").write_bytes(metadata)
    (output_dir / "manifest.json").write_text(
        json.dumps({"metadata_sha256": hashlib.sha256(metadata).hexdigest()}),
        encoding="utf-8",
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            [
                {
                    "scene_name": scene,
                    "fold_8_kmeans_16_input": list(split_indices),
                    "fold_8_kmeans_32_input": list(split_indices),
                }
            ]
        ),
        encoding="utf-8",
    )
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    required_names = {frame_names[index] for index in split_indices}
    required_names.update(frame_names[::8])
    required_names.add("transforms.json")
    required_size = len(transform_payload) + sum(
        len(f"payload-{all_image_names.index(name)}".encode())
        for name in required_names
        if name != "transforms.json"
    )
    physical_names = set(physical_images) | {"transforms.json"}
    physical_size = len(transform_payload) + sum(
        len(f"payload-{all_image_names.index(name)}".encode())
        for name in physical_names
        if name != "transforms.json"
    )
    status_entries = []
    for name in physical_images:
        if name not in required_names:
            continue
        image = image_by_name[name]
        payload = f"payload-{all_image_names.index(name)}".encode()
        status_entries.append(
            {
                "repo_path": image.path,
                "status": "reused",
                "expected_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    status_entries.append(
        {
            "repo_path": transform.path,
            "status": "reused",
            "expected_bytes": len(transform_payload),
            "sha256": hashlib.sha256(transform_payload).hexdigest(),
        }
    )
    (output_dir / "status.json").write_text(
        json.dumps({"status": "complete", "entries": status_entries}),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_script, "EXPECTED_PROTOCOL_SHA256", protocol_sha256)
    monkeypatch.setattr(verify_script, "EXPECTED_FILE_COUNT", len(required_names))
    monkeypatch.setattr(verify_script, "EXPECTED_TOTAL_BYTES", required_size)
    monkeypatch.setattr(
        verify_script,
        "EXPECTED_RETAINED_EXTRA_FILES",
        len(physical_names - required_names),
    )
    monkeypatch.setattr(
        verify_script,
        "EXPECTED_RETAINED_EXTRA_BYTES",
        physical_size - required_size,
    )
    monkeypatch.setattr(verify_script, "EXPECTED_PHYSICAL_FILE_COUNT", len(physical_names))
    monkeypatch.setattr(verify_script, "EXPECTED_PHYSICAL_BYTES", physical_size)
    monkeypatch.setattr(verify_script, "load_inventory_artifacts", lambda _: inventory)
    args = SimpleNamespace(
        inventory_dir=output_dir,
        output=output_dir / "completion_verification.json",
        data_root=data_root,
        protocol_json=protocol_path,
        workers=1,
        target_every=8,
    )
    return args, {"scene": scene, "images": image_by_name}


def test_verifier_run_accepts_independent_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    assert verify_script.run(args) == 0
    result = json.loads(args.output.read_text(encoding="utf-8"))
    assert result["status"] == "verified"
    assert result["file_count"] == 2
    assert result["total_bytes"] == result["physical_bytes"]
    assert result["retained_extra_files"] == 0
    assert result["source_hash_verified"]["all_entries"] is True
    assert result["protocol_coverage"] == {
        "status": "complete",
        "mapping": "transforms_frame_order",
        "protocols": ["full16", "ar32"],
        "required_files": 2,
        "missing_files": 0,
    }
    assert len(result["entries"]) == 2
    assert result["extra_entries"] == []


def test_verifier_hashes_retained_inventory_extras_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, fixture = _fixture(
        tmp_path,
        monkeypatch,
        frame_names=("frame_00001.png", "frame_00002.png"),
        physical_images=("frame_00001.png", "frame_00002.png"),
    )
    assert verify_script.run(args) == 0
    result = json.loads(args.output.read_text(encoding="utf-8"))
    assert result["file_count"] == 2
    assert result["retained_extra_files"] == 1
    assert result["physical_count"] == 3
    assert len(result["entries"]) == 2
    assert len(result["extra_entries"]) == 1
    assert result["extra_entries"][0]["repo_path"].endswith("frame_00002.png")
    assert result["source_hash_verified"]["extra_entry_count"] == 1
    assert fixture["images"]["frame_00002.png"].path not in {
        entry["repo_path"] for entry in result["entries"]
    }


def test_old_index_plus_one_selection_fails_real_frame_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The old selector would choose frame_00002 for protocol index 1.  All three
    # physical files below have valid pinned identities, but frame_00003 is the
    # required transforms-order target and is absent.
    args, _ = _fixture(
        tmp_path,
        monkeypatch,
        frame_names=("frame_00001.png", "frame_00003.png", "frame_00005.png"),
        inventory_extra_names=("frame_00002.png",),
        split_indices=(1,),
        physical_images=("frame_00001.png", "frame_00002.png"),
    )
    with pytest.raises(ValueError, match="required protocol files are missing"):
        verify_script.run(args)


def test_verifier_run_rejects_tampered_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, fixture = _fixture(tmp_path, monkeypatch)
    image_path = args.data_root / fixture["images"]["frame_00001.png"].path
    image_path.write_bytes(b"tampered file")
    with pytest.raises(ValueError, match="size mismatch|source identity mismatch"):
        verify_script.run(args)
