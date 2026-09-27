# Pinned Long-LRM split test fixture

Unmodified scene names and frame-index metadata from Long-LRM commit `0e89431a964b8a7e2fdec24e3ca5ec74a26dcafc`; exact URL and SHA256 are retained in the adjacent provenance JSON. This is the public 140-scene split descriptor, not dataset images, poses, depth, or weights. The loader verifies its pinned hash, and tests retain the tampered-bytes rejection check.

The upstream Apache-2.0 LICENSE is included. This fixture makes metadata contract tests self-contained without reading a developer's ignored outputs directory. No benchmark media or protected evaluation split is opened by these tests.
