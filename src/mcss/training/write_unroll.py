"""Differentiable offline trajectory unrolling without query supervision."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.feedback import anchored_observation
from mcss.dynamic.types import Action, FastWeights, OnlineObservation
from mcss.dynamic.write_rule import commit_write
from mcss.types import SceneState


def unroll_observed_episode(
    carrier,
    write_rule,
    warmup: Sequence[OnlineObservation],
    stream: Sequence[OnlineObservation],
    actions: Sequence[Action],
    episode_id: str,
) -> tuple[SceneState, FastWeights, ObservationCache, Tensor]:
    """Unroll one observed-only episode while retaining the complete gradient graph.

    ``warmup`` and ``stream`` are raw world-coordinate observations.  The first warmup
    camera fixes the shared local frame, and every observation is rebased through that
    same camera before it enters the carrier.  This intentionally has no renderer,
    labels, policy, or ``torch.no_grad`` boundary.
    """

    warmup = tuple(warmup)
    stream = tuple(stream)
    actions = tuple(Action(action) for action in actions)
    _validate_episode_inputs(warmup, stream, actions, episode_id)

    scene_id = warmup[0].scene_id
    anchor_c2w = warmup[0].camera.c2w
    cache = ObservationCache(episode_id=episode_id, scene_id=scene_id)
    fast = carrier.initial_fast(episode_id)

    for observation in warmup:
        arrived = anchored_observation(observation, anchor_c2w)
        cache = cache.append(arrived, carrier.encode(arrived))

    state = carrier.materialize(cache, fast)
    for observation, action in zip(stream, actions, strict=True):
        arrived = anchored_observation(observation, anchor_c2w)
        cache = cache.append(arrived, carrier.encode(arrived))
        if action != Action.OFF:
            # ALL is committed from this one trace, so both matrices see identical old weights.
            proposal = write_rule.propose(carrier.trace(cache, fast))
            fast = commit_write(fast, proposal, action, cache_revision=cache.revision)
        # OFF deliberately still rebuilds the state after the arrived observation is cached.
        state = carrier.materialize(cache, fast)

    return state, fast, cache, anchor_c2w.clone()


def _validate_episode_inputs(
    warmup: tuple[OnlineObservation, ...],
    stream: tuple[OnlineObservation, ...],
    actions: tuple[Action, ...],
    episode_id: str,
) -> None:
    if not episode_id:
        raise ValueError("episode_id must be nonempty")
    if not warmup:
        raise ValueError("warmup must contain the anchor observation")
    if len(actions) != len(stream):
        raise ValueError("actions must have exactly one action per stream observation")

    scene_id = warmup[0].scene_id
    image_size = warmup[0].camera.image_size
    device = warmup[0].rgb.device
    dtype = warmup[0].rgb.dtype
    previous_frame_id = -1
    for observation in (*warmup, *stream):
        if observation.scene_id != scene_id:
            raise ValueError("one unroll cannot mix scene identities")
        if observation.camera.image_size != image_size:
            raise ValueError("one unroll requires a shared image size")
        if observation.rgb.device != device or observation.rgb.dtype != dtype:
            raise ValueError("one unroll requires a shared RGB device and dtype")
        if observation.frame_id <= previous_frame_id:
            raise ValueError("observations must have strictly increasing frame IDs")
        previous_frame_id = observation.frame_id
