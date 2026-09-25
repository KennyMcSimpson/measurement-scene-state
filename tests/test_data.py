import torch

from mcss.data.collate import collate_scene_examples
from mcss.data.synthetic import SyntheticSceneDataset


def test_synthetic_dataset_is_deterministic_and_physically_typed() -> None:
    dataset = SyntheticSceneDataset(
        length=3, image_size=(12, 16), context_views=2, target_views=1, seed=19
    )

    first = dataset[1]
    repeated = dataset[1]

    torch.testing.assert_close(first.context_rgb, repeated.context_rgb)
    torch.testing.assert_close(first.target_depth, repeated.target_depth)
    assert first.context_rgb.shape == (2, 3, 12, 16)
    assert first.target_depth.shape == (1, 1, 12, 16)
    assert first.target_normal.shape == (1, 3, 12, 16)
    assert first.target_visibility.shape == (1, 1, 12, 16)
    visible = first.target_visibility[:, 0].bool()
    normal_values = first.target_normal.permute(0, 2, 3, 1)[visible]
    torch.testing.assert_close(
        torch.linalg.vector_norm(normal_values, dim=-1),
        torch.ones(normal_values.shape[0]),
        atol=1e-4,
        rtol=1e-4,
    )
    assert not torch.equal(first.context_cameras.c2w[0], first.target_cameras.c2w[0])


def test_synthetic_examples_collate_into_scene_batch() -> None:
    dataset = SyntheticSceneDataset(
        length=2, image_size=(8, 8), context_views=2, target_views=1, seed=7
    )

    batch = collate_scene_examples([dataset[0], dataset[1]])

    assert batch.context_rgb.shape == (2, 2, 3, 8, 8)
    assert batch.target_rgb.shape == (2, 1, 3, 8, 8)
    assert batch.bounds.shape == (2, 2, 3)
    assert batch.context_cameras.leading_shape == (2, 2)
