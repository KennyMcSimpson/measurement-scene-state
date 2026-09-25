import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from requests.structures import CaseInsensitiveDict

import mcss.data.dl3dv_download as dl3dv_download
from mcss.data.dl3dv_download import (
    EXPECTED_SCENE_COUNT,
    REVISION,
    DownloadBusy,
    ExclusiveLock,
    IntegrityError,
    Inventory,
    RemoteFile,
    SceneMeta,
    _fetch_bytes,
    git_blob_sha1,
    hf_resolve_url,
    parse_benchmark_metadata,
    protocol_frame_selection,
    select_files,
    verify_local_file,
)


def _scene(value: str = "a" * 64) -> SceneMeta:
    return SceneMeta(0, value, {"scene": "Synthetic", "environment": "bounded"})


def _remote(path: str, payload: bytes, *, lfs: bool = False) -> RemoteFile:
    sha256 = hashlib.sha256(payload).hexdigest()
    size = len(payload)
    prefix = f"blob {size}\0".encode("ascii")
    blob = hashlib.sha1(prefix + payload).hexdigest()
    return RemoteFile(
        path=path,
        size=size,
        git_blob_id=blob,
        lfs_sha256=sha256 if lfs else None,
        lfs_size=size if lfs else None,
        scene_hash=path.split("/", 1)[0] if len(path.split("/", 1)[0]) == 64 else None,
    )


def _inventory(scene: str = "a" * 64) -> Inventory:
    metadata = (
        b",hash,scene\n0," + scene.encode("ascii") + b",Synthetic\n"
    )
    transforms = _remote(f"{scene}/nerfstudio/transforms.json", b"{}", lfs=False)
    cameras = _remote(f"{scene}/nerfstudio/colmap/sparse/0/cameras.bin", b"camera", lfs=True)
    frame1 = _remote(f"{scene}/nerfstudio/images_4/frame_00001.png", b"frame-1", lfs=True)
    frame2 = _remote(f"{scene}/nerfstudio/images_4/frame_00002.png", b"frame-2", lfs=True)
    metadata_file = RemoteFile("benchmark-meta.csv", len(metadata), "a" * 40, None, None, None)
    return Inventory(
        metadata,
        metadata_file,
        (_scene(scene),),
        (transforms, cameras, frame1, frame2),
    )


def _protocol_fixture(
    tmp_path: Path,
    *,
    frame_paths: tuple[str, ...] = (
        "camera/frame_00007.png",
        "camera/frame_00011.png",
        "camera/frame_00019.png",
        "camera/frame_00023.png",
        "camera/frame_00031.png",
    ),
    inventory_names: tuple[str, ...] | None = None,
) -> tuple[str, Inventory, Path]:
    scene = "a" * 64
    transforms_payload = json.dumps(
        {"frames": [{"file_path": path} for path in frame_paths]},
        separators=(",", ":"),
    ).encode("utf-8")
    transforms = _remote(f"{scene}/nerfstudio/transforms.json", transforms_payload)
    names = inventory_names
    if names is None:
        names = tuple(dict.fromkeys(path.replace("\\", "/").split("/")[-1] for path in frame_paths))
    images = tuple(
        _remote(
            f"{scene}/nerfstudio/images_4/{name}",
            name.encode("ascii"),
            lfs=True,
        )
        for name in names
    )
    metadata = f"hash,scene\n{scene},Synthetic\n".encode("ascii")
    metadata_file = RemoteFile("benchmark-meta.csv", len(metadata), "a" * 40, None, None, None)
    inventory = Inventory(
        metadata,
        metadata_file,
        (_scene(scene),),
        (transforms, *images),
    )
    data_root = tmp_path / "data"
    transform_path = data_root / scene / "nerfstudio" / "transforms.json"
    transform_path.parent.mkdir(parents=True)
    transform_path.write_bytes(transforms_payload)
    return scene, inventory, data_root


def test_parse_metadata_rejects_untrusted_shape_and_duplicate_hashes() -> None:
    scene = "a" * 64
    payload = ("index,hash,scene\n0," + scene + ",Synthetic\n").encode()
    assert parse_benchmark_metadata(payload, expected_count=1)[0].hash == scene
    with pytest.raises(ValueError, match="duplicate"):
        parse_benchmark_metadata(
            ("index,hash\n0," + scene + "\n1," + scene + "\n").encode(),
            expected_count=None,
        )
    with pytest.raises(ValueError, match="invalid scene hash"):
        parse_benchmark_metadata(b"index,hash\n0,../escape\n", expected_count=None)


def test_metadata_requires_the_140_scene_contract() -> None:
    with pytest.raises(ValueError, match="140"):
        parse_benchmark_metadata(b"hash\n" + b"a" * 64 + b"\n")
    assert EXPECTED_SCENE_COUNT == 140


def test_selection_is_canonical_scene_and_frame_bounded(tmp_path: Path) -> None:
    inventory = _inventory()
    selected = select_files(inventory, tmp_path, include_images4=True, frame_ids=[2])
    assert [item.remote.kind for item in selected] == ["transforms", "images_4"]
    assert selected[0].remote.relative_path == "nerfstudio/transforms.json"
    assert selected[1].remote.relative_path.endswith("frame_00002.png")
    assert all(str(item.destination).startswith(str(tmp_path)) for item in selected)

    with pytest.raises(ValueError, match="unknown"):
        select_files(inventory, tmp_path, scenes=["b" * 64])


def test_protocol_selection_uses_transform_positions_and_target_stride(tmp_path: Path) -> None:
    scene, inventory, data_root = _protocol_fixture(tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        '[{"scene_name": "' + scene + '", '
        '"fold_8_kmeans_16_input": [1], '
        '"fold_8_kmeans_32_input": [3]}]',
        encoding="utf-8",
    )
    selected = protocol_frame_selection(
        protocol,
        inventory,
        data_root=data_root,
        target_every=2,
    )
    assert selected[scene] == {7, 11, 19, 23, 31}


def test_protocol_selection_rejects_out_of_range_index(tmp_path: Path) -> None:
    scene, inventory, data_root = _protocol_fixture(tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            [{
                "scene_name": scene,
                "fold_8_kmeans_16_input": [5],
                "fold_8_kmeans_32_input": [],
            }]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="out of range"):
        protocol_frame_selection(protocol, inventory, data_root=data_root)


def test_protocol_selection_rejects_duplicate_frame_paths(tmp_path: Path) -> None:
    scene, inventory, data_root = _protocol_fixture(
        tmp_path,
        frame_paths=("camera/frame_00007.png", "other/frame_00007.png"),
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps([{
            "scene_name": scene,
            "fold_8_kmeans_16_input": [0],
            "fold_8_kmeans_32_input": [],
        }]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate frame path"):
        protocol_frame_selection(protocol, inventory, data_root=data_root)


def test_protocol_selection_rejects_missing_inventory_target(tmp_path: Path) -> None:
    scene, inventory, data_root = _protocol_fixture(
        tmp_path,
        inventory_names=("frame_00007.png",),
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps([{
            "scene_name": scene,
            "fold_8_kmeans_16_input": [0],
            "fold_8_kmeans_32_input": [],
        }]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing inventory target"):
        protocol_frame_selection(protocol, inventory, data_root=data_root)


def test_protocol_selection_verifies_local_transforms_identity(tmp_path: Path) -> None:
    scene, inventory, data_root = _protocol_fixture(tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps([{
            "scene_name": scene,
            "fold_8_kmeans_16_input": [0],
            "fold_8_kmeans_32_input": [],
        }]),
        encoding="utf-8",
    )
    transform_path = data_root / scene / "nerfstudio" / "transforms.json"
    transform_path.write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="mismatch"):
        protocol_frame_selection(protocol, inventory, data_root=data_root)


def test_verify_local_file_checks_lfs_and_git_blob_identity(tmp_path: Path) -> None:
    payload = b"verified fixture"
    path = tmp_path / "file.bin"
    path.write_bytes(payload)
    remote_lfs = _remote("a" * 64 + "/nerfstudio/file.bin", payload, lfs=True)
    result = verify_local_file(path, remote_lfs)
    assert result["bytes"] == len(payload)
    remote_git = _remote("a" * 64 + "/nerfstudio/file.bin", payload, lfs=False)
    assert git_blob_sha1(path) == remote_git.git_blob_id
    path.write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="size mismatch|SHA1"):
        verify_local_file(path, remote_git)


def test_hf_url_is_revision_pinned_and_path_escaped() -> None:
    remote = _remote("a" * 64 + "/nerfstudio/images_4/frame_00001.png", b"x", lfs=True)
    url = hf_resolve_url(remote)
    assert REVISION in url
    assert "/datasets/DL3DV/DL3DV-Benchmark/resolve/" in url
    with pytest.raises(ValueError):
        hf_resolve_url("../outside")


def test_fetch_bytes_verifies_remote_identity() -> None:
    payload = b"network fixture"
    remote = _remote("a" * 64 + "/nerfstudio/file.bin", payload, lfs=True)

    class Response:
        status_code = 200
        url = "https://fixture"
        headers = CaseInsensitiveDict({"Content-Length": str(len(payload))})
        content = payload

        def raise_for_status(self):
            return None

        def close(self):
            return None

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    assert _fetch_bytes(remote, token="fixture", session=Session()) == payload


def test_exclusive_lock_rejects_second_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "download.lock"
    with ExclusiveLock(lock_path):
        with pytest.raises(DownloadBusy):
            with ExclusiveLock(lock_path):
                pass


def test_download_selected_uses_bounded_workers_and_atomic_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _inventory()
    selected = select_files(inventory, tmp_path / "data", include_images4=True)
    selected = selected[:2]
    barrier = threading.Barrier(2)
    thread_names: set[str] = set()

    def fake_download_one(item, **kwargs):
        thread_names.add(threading.current_thread().name)
        barrier.wait(timeout=5)
        time.sleep(0.01)
        return {"status": "downloaded", "bytes": item.remote.size}

    monkeypatch.setattr(dl3dv_download, "_download_one", fake_download_one)
    status_path = tmp_path / "status.json"
    result = dl3dv_download.download_selected(
        selected,
        token="fixture",
        status_path=status_path,
        lock_path=tmp_path / "download.lock",
        reserve_bytes=0,
        workers=2,
    )

    assert result["status"] == "complete"
    assert len(thread_names) == 2
    assert [entry["status"] for entry in result["entries"]] == [
        "downloaded",
        "downloaded",
    ]
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "complete"


def test_remote_file_rejects_cache_and_unsafe_paths() -> None:
    with pytest.raises(ValueError, match="cache"):
        RemoteFile.from_hf(SimpleNamespace(path=".cache/filelist.bin", size=1, blob_id="a" * 40))
    with pytest.raises(ValueError, match="unsafe"):
        RemoteFile.from_hf(SimpleNamespace(path="../escape", size=1, blob_id="a" * 40))
