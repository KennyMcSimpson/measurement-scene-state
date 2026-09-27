# Third-party material

The project source in `src/` and project scripts are covered by the root
`LICENSE` unless a file says otherwise. The following paths are pinned upstream
Git repositories used by local experiments. They are recorded as Git links so
this repository does not relicense or vendor third-party source:

- `third_party/dinov2/` — DINOv2, commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- `third_party/map-anything/` — MapAnything, commit `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`.
- `third_party/unidepth/` — UniDepth, commit `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`.
- `third_party/emvsnet/` — EMVSNet, commit `982bf4c57f00b2d389cc7c97aae1df70d6ccd6f6`.

Clone them with `git submodule update --init --recursive` after cloning this
repository. The local EMVSNet working-tree adjustment used by one historical
probe is recorded in `docs/provenance/artifacts/emvsnet_local_changes.patch`;
it is not presented as upstream code.

Pretrained weights, downloaded datasets, Hugging Face caches and experiment
checkpoints are not included in this public repository. See
`docs/provenance/DATA.md` for source URLs and local hashes.

The PDF in `research/source/` is Kenny's supplied ICLR draft snapshot. Its
publication status and any coauthor rights are not asserted by this repository.

## Long-LRM metadata test fixture

`tests/fixtures/long_lrm/` contains the unmodified public scene/frame-index split descriptor from Long-LRM revision `0e89431a964b8a7e2fdec24e3ca5ec74a26dcafc`, with source URL/hash provenance and upstream Apache-2.0 LICENSE. It contains no dataset images or weights. See its README.
