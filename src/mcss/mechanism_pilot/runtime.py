"""Isolated mechanism branches around the unchanged dynamic carrier.

No query readers, labels, training, or controller optimization belong in this module.
The caller must evaluate only sealed states through the independent evaluator.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter

import torch

from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.feedback import anchored_observation, preupdate_feedback
from mcss.dynamic.types import (
    ACTION_ORDER,
    Action,
    FastWeights,
    OnlineObservation,
    SealedScene,
    clone_scene_state,
    hash_fast_content,
    hash_scene_state,
    hash_value,
)
from mcss.dynamic.write_rule import commit_write
from mcss.types import SceneState


@dataclass(frozen=True)
class FrameRoles:
    context_a: tuple[int, ...]
    context_b: tuple[int, ...]
    stream: tuple[int, ...]
    continuation: tuple[int, ...]
    sealed_query: tuple[int, ...]

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            ids = tuple(getattr(self, name))
            if not ids or len(set(ids)) != len(ids) or any(i < 0 for i in ids):
                raise ValueError(f"Invalid frame IDs for {name}")
            object.__setattr__(self, name, ids)
        if len(self.context_a) != len(self.context_b):
            raise ValueError("Context sizes must match")
        if self.context_a[0] != self.context_b[0]:
            raise ValueError("Contexts require the same first coordinate anchor")
        if set(self.context_a) & set(self.context_b) != {self.context_a[0]}:
            raise ValueError("Contexts may share only their coordinate anchor")
        groups = [
            set(self.context_a) | set(self.context_b),
            set(self.stream),
            set(self.continuation),
            set(self.sealed_query),
        ]
        if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
            raise ValueError("Context, stream, continuation, and query roles overlap")

    def require_visible(self, observation):
        if type(observation) is not OnlineObservation:
            raise TypeError("Only RGB/camera OnlineObservation is allowed; no depth or labels")
        allowed = set(self.context_a + self.context_b + self.stream + self.continuation)
        if observation.frame_id not in allowed:
            raise ValueError("Sealed query or undeclared frame cannot enter runtime")


@dataclass(frozen=True)
class Branch:
    cache: ObservationCache
    fast: FastWeights
    state: SceneState
    anchor_c2w: torch.Tensor
    history: tuple[dict, ...] = ()

    @property
    def summary(self):
        """Derived on access, so no stale cached history/state summary can survive a write."""
        return {
            "observed_count": self.cache.revision,
            "observed_ids": [e.observation.frame_id for e in self.cache.entries],
            "fast_step": self.fast.step,
            "fast_fuse_norm": float(self.fast.delta_fuse.norm()),
            "fast_complete_norm": float(self.fast.delta_complete.norm()),
            "density_mean": float(self.state.density_logits.mean()),
            "previous_action": self.history[-1]["action"] if self.history else "OFF",
            "previous_write_magnitude": self.history[-1]["write_magnitude"]
            if self.history
            else 0.0,
            "state_hash": hash_scene_state(self.state),
            "fast_hash": hash_fast_content(self.fast),
        }

    def fork(self):
        cache = ObservationCache(self.cache.episode_id, self.cache.scene_id)
        for entry in self.cache.entries:
            cache = cache.append(entry.observation, entry.features)
        return Branch(
            cache,
            self.fast.fork(),
            clone_scene_state(self.state),
            self.anchor_c2w.detach().clone(),
            deepcopy(self.history),
        )


class MechanismRuntime:
    def __init__(self, carrier, writer, renderer, roles: FrameRoles):
        self.carrier, self.writer, self.renderer, self.roles = carrier, writer, renderer, roles
        for module in (carrier, writer, renderer):
            module.eval().requires_grad_(False)
        if list(renderer.parameters()):
            raise ValueError("Fixed measurement must have no learned parameters")
        self.checkpoint_hash = hash_value(
            {"carrier": carrier.state_dict(), "writer": writer.state_dict()}
        )
        self.config_hash = hash_value(
            {
                "carrier": asdict(carrier.config),
                "writer": asdict(writer.write_config),
                "render_samples": renderer.n_samples,
                "roles": asdict(roles),
            }
        )

    @torch.no_grad()
    def build(self, context, *, episode_id, anchor_c2w=None):
        context = tuple(context)
        if not context:
            raise ValueError("Empty context")
        for observation in context:
            self.roles.require_visible(observation)
        ids = tuple(o.frame_id for o in context)
        if ids not in (self.roles.context_a, self.roles.context_b, (self.roles.context_a[0],)):
            raise ValueError("State construction requires a declared complete context or anchor")
        anchor = context[0].camera.c2w.detach().clone()
        if anchor_c2w is not None and not torch.equal(anchor, anchor_c2w):
            raise ValueError("The common anchor must match its declared camera")
        cache = ObservationCache(episode_id, context[0].scene_id)
        fast = self.carrier.initial_fast(episode_id)
        for observation in context:
            arrived = anchored_observation(observation, anchor)
            cache = cache.append(arrived, self.carrier.encode(arrived))
        return Branch(cache, fast, self.carrier.materialize(cache, fast), anchor)

    def _arrival(self, branch, observation):
        self.roles.require_visible(observation)
        if observation.scene_id != branch.cache.scene_id:
            raise ValueError("Continuation scene mismatch")
        if observation.frame_id <= branch.cache.entries[-1].observation.frame_id:
            raise ValueError("Continuation must advance frame order")
        return anchored_observation(observation, branch.anchor_c2w)

    @torch.no_grad()
    def preview(self, branch, observation):
        """Feedback from old state, before any cache append; RGB/camera only."""
        arrived = self._arrival(branch, observation)
        return preupdate_feedback(
            branch.state,
            arrived,
            self.renderer,
            image_feature=self.carrier.encode(arrived),
            previous_camera=branch.cache.entries[-1].observation.camera,
        )

    @torch.no_grad()
    def step(self, branch, observation, action):
        action = Action(action)
        arrived = self._arrival(branch, observation)
        if observation.frame_id not in self.roles.stream + self.roles.continuation:
            raise ValueError("A write step requires a declared stream/continuation frame")
        started = perf_counter()
        child = branch.fork()
        features = self.carrier.encode(arrived)
        feedback = preupdate_feedback(
            child.state,
            arrived,
            self.renderer,
            image_feature=features,
            previous_camera=child.cache.entries[-1].observation.camera,
        )
        pre_summary = child.summary
        cache = child.cache.append(arrived, features)  # OFF receives identical evidence.
        fast = child.fast
        candidate_hash = hash_value(self.carrier._candidate_ids)
        if action != Action.OFF:
            trace = self.carrier.trace(cache, fast)
            candidate_hash = hash_value(trace.candidate_ids)
            proposal = self.writer.propose(trace)  # BOTH proposals use the same old trace.
            fast = commit_write(fast, proposal, action, cache_revision=cache.revision)
        magnitude = float(
            torch.sqrt(
                (fast.delta_fuse - child.fast.delta_fuse).square().sum()
                + (fast.delta_complete - child.fast.delta_complete).square().sum()
            )
        )
        state = self.carrier.materialize(cache, fast)  # Fresh read even at the final step.
        record = {
            "frame_id": observation.frame_id,
            "action": str(action),
            "observation_hash": hash_value(observation),
            "candidate_pack_hash": candidate_hash,
            "pre_summary": pre_summary,
            "feedback": {
                k: v.detach().cpu().reshape(-1).tolist() if isinstance(v, torch.Tensor) else v
                for k, v in feedback.items()
                if k != "image_feature"
            },
            "write_magnitude": magnitude,
            "seconds": perf_counter() - started,
            "operations": {
                "encode": 1,
                "feedback_render": 1,
                "materialize": 1,
                "trace": int(action != Action.OFF),
                "proposal_pair": int(action != Action.OFF),
            },
        }
        return Branch(cache, fast, state, child.anchor_c2w, (*child.history, record))

    def matched_histories(self, context, stream, *, episode_id):
        stream = tuple(stream)
        if len(stream) != 2 or tuple(o.frame_id for o in stream) != self.roles.stream:
            raise ValueError("Matched histories require the same declared two-frame stream")
        initial = self.build(context, episode_id=episode_id)
        histories = {}
        for name, actions in (
            ("FC", (Action.FUSE, Action.COMPLETE)),
            ("CF", (Action.COMPLETE, Action.FUSE)),
        ):
            branch = initial.fork()
            for observation, action in zip(stream, actions, strict=True):
                branch = self.step(branch, observation, action)
            histories[name] = branch
        return histories

    def continuation_branches(self, histories, observation):
        if observation.frame_id not in self.roles.continuation:
            raise ValueError("Undeclared continuation")
        return {
            name: {str(action): self.step(branch, observation, action) for action in ACTION_ORDER}
            for name, branch in histories.items()
        }

    def seal(self, branch, *, split_id="synthetic", query_vault_id="synthetic"):
        state = clone_scene_state(branch.state)
        return SealedScene(
            branch.cache.episode_id,
            branch.cache.scene_id,
            split_id,
            query_vault_id,
            state,
            hash_scene_state(state),
            tuple(e.observation.frame_id for e in branch.cache.entries),
            self.checkpoint_hash,
            self.config_hash,
            hash_fast_content(branch.fast),
            branch.anchor_c2w.detach().clone(),
        )
