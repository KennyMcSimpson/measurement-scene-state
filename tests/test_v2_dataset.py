import pytest

from mcss.vision_probe.v2_access import AccessDenied
from mcss.vision_probe.v2_dataset import (
    SUPPORTS_REMOTE_TARGET_FETCH,
    RangeZip,
    fetch_target_masks,
    select_metadata,
)


def test_unverified_clip_identity_cannot_be_promoted():
    with pytest.raises(ValueError, match="UNVERIFIED_ORIGINAL_VIDEO_ID"):
        select_metadata([], ["1184_cut_chilli"], {})


def test_target_fetch_currently_fail_closed_without_network():
    assert SUPPORTS_REMOTE_TARGET_FETCH is False
    with pytest.raises(AccessDenied, match="NO_QUALIFIED_DATASET_ADAPTER"):
        fetch_target_masks({"independence_status": "VERIFIED"}, "unused", "unused", guard=None)


def test_range_budget_checked_before_request(monkeypatch):
    with RangeZip(budget=100) as reader:

        def forbidden(*args, **kwargs):
            raise AssertionError("No network should be attempted")

        monkeypatch.setattr(reader.session, "get", forbidden)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            reader.read(101)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            reader.read()


def test_server_ignoring_range_rejected_without_consuming_body(monkeypatch):
    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            raise AssertionError("Full archive response body must not be consumed")

    with RangeZip() as reader:
        monkeypatch.setattr(reader.session, "get", lambda *args, **kwargs: Response())
        with pytest.raises(RuntimeError, match="exact bounded byte range"):
            reader.read(22)
        assert reader.downloaded_bytes == 0
