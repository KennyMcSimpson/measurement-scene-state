import zipfile

import numpy as np
import pytest

from mcss.mechanism_pilot.data_prep import check_geometry, resize_rgb_depth, select_members


def infos(scene="ai_001_001"):
    names = [f"{scene}/_detail/metadata_scene.csv"] + [
        f"{scene}/_detail/cam_00/camera_keyframe_{kind}.hdf5"
        for kind in ("frame_indices", "positions", "orientations")
    ]
    for fid in range(2, 20):
        names.extend(
            [
                f"{scene}/images/scene_cam_00_final_preview/frame.{fid:04d}.color.jpg",
                f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5",
            ]
        )
    names.append(f"{scene}/images/scene_cam_00_geometry_hdf5/frame.0002.position.hdf5")
    result = [zipfile.ZipInfo(n) for n in names]
    for info in result:
        info.CRC = 0
    return result


def test_metadata_selection_uses_actual_common_ids_and_rejects_other_splits():
    plan = select_members("ai_001_001", infos())
    assert plan["frame_ids"] == list(range(2, 18))
    assert not any("normal" in n for n in plan["members"])
    with pytest.raises(ValueError, match="whitelist"):
        select_members("ai_004_003", infos("ai_004_003"))


def test_budget_and_insufficient_frames_rejected():
    members = infos()
    members[0].file_size = 1024**3 + 1
    with pytest.raises(ValueError, match="budget"):
        select_members("ai_001_001", members)
    with pytest.raises(ValueError, match="Insufficient"):
        select_members("ai_001_001", infos()[:8])


def test_masked_depth_resize_does_not_spread_nan_or_zero_bias():
    depth = np.array([[2.0, np.nan], [0.0, 2.0]])
    rgb = np.zeros((2, 2, 3), np.uint8)
    resized_rgb, resized_depth = resize_rgb_depth(rgb, depth, (1, 1))
    assert resized_rgb.shape == (1, 1, 3)
    np.testing.assert_allclose(resized_depth, 2.0)


def test_geometry_detects_wrong_scale_or_camera_axes():
    depth = np.ones((3, 4)) * 2
    y, x = np.indices(depth.shape)
    rays = np.stack([x, y, np.ones_like(x)], -1).astype(float)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    positions = rays * depth[..., None]
    report = check_geometry("fixture", depth, positions, np.eye(4), np.eye(3), 1)
    assert report["p95_error_m"] < 1e-12
    with pytest.raises(ValueError, match="calibration failed"):
        check_geometry("fixture", depth, positions, np.eye(4), np.eye(3), 2)


class Response:
    def __init__(self, status, headers, content=b"abc"):
        self.status_code = status
        self.headers = headers
        self.content = content
        self.body_read = False

    def raise_for_status(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        self.body_read = True
        yield self.content


class Session:
    def __init__(self, response):
        self.response = response
        self.request_count = 0

    def head(self, *args, **kwargs):
        return Response(200, {"Content-Length": "10000000000", "ETag": '"stable"'})

    def get(self, *args, **kwargs):
        assert kwargs["stream"] is True
        self.request_count += 1
        return self.response


def test_full_archive_fallback_rejected_before_reading_body():
    from mcss.mechanism_pilot.data_prep import BoundedRangeReader

    response = Response(200, {"ETag": '"stable"'})
    reader = BoundedRangeReader("fixture", Session(response))
    with pytest.raises(OSError, match="full ZIP fallback prohibited"):
        reader.read(3)
    assert not response.body_read


def test_global_budget_refuses_request_before_network():
    from mcss.mechanism_pilot.data_prep import BoundedRangeReader

    session = Session(Response(206, {}))
    reader = BoundedRangeReader("fixture", session)
    reader.shared_budget = [2 * 1024**3 - 2]
    with pytest.raises(OSError, match="Shared transfer budget"):
        reader.read(3)
    assert session.request_count == 0


def test_exact_range_rejects_changed_identity_before_body():
    from mcss.mechanism_pilot.data_prep import BoundedRangeReader

    response = Response(206, {"ETag": '"changed"', "Content-Range": "bytes 0-2/10000000000"})
    reader = BoundedRangeReader("fixture", Session(response))
    with pytest.raises(OSError, match="changed"):
        reader.read(3)
    assert not response.body_read
