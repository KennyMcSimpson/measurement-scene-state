"""Causal single-episode deployment orchestration; no label or teacher imports."""

from dataclasses import asdict
from time import perf_counter

import torch

from mcss.dynamic.budget import WorkBudget
from mcss.dynamic.cache import ObservationCache
from mcss.dynamic.feedback import anchored_observation, preupdate_feedback
from mcss.dynamic.types import (
    Action,
    ControlInput,
    SealedScene,
    clone_scene_state,
    hash_fast_content,
    hash_scene_state,
    hash_value,
)
from mcss.dynamic.write_rule import commit_write


class StreamingRunner:
    def __init__(self, carrier, write_rule, renderer, policy, *, max_units=1e12):
        self.carrier = carrier
        self.write_rule = write_rule
        self.renderer = renderer
        self.policy = policy
        self.max_units = max_units
        self.active = False
        for module in (carrier, write_rule, renderer):
            module.eval()
            module.requires_grad_(False)
        if isinstance(policy, torch.nn.Module):
            policy.eval()
            policy.requires_grad_(False)
        if list(renderer.parameters()):
            raise ValueError("Dynamic main method requires a parameter-free renderer")
        self.checkpoint_hash = hash_value(
            {"carrier": carrier.state_dict(), "write_rule": write_rule.state_dict()}
        )

    @torch.no_grad()
    def reset(
        self, warmup, *, episode_id, scene_id, split_id, query_vault_id, stream_steps, query_count=4
    ):
        self.active = False
        if not warmup or stream_steps < 0 or query_count < 0:
            raise ValueError("Invalid episode protocol")
        if any(observation.scene_id != scene_id for observation in warmup):
            raise ValueError("Warmup scene identity mismatch")
        image_size = warmup[0].camera.image_size
        if any(observation.camera.image_size != image_size for observation in warmup):
            raise ValueError("All observations must share declared resolution")
        self.anchor_c2w = warmup[0].camera.c2w.detach().clone()
        self.episode_id, self.scene_id = episode_id, scene_id
        self.split_id, self.query_vault_id = split_id, query_vault_id
        self.remaining_steps, self.query_count = stream_steps, query_count
        self.image_size = image_size
        self.cache = ObservationCache(episode_id, scene_id)
        self.fast = self.carrier.initial_fast(episode_id)
        self.budget = WorkBudget(
            self.carrier.config,
            image_size,
            self.renderer.n_samples,
            self.max_units,
            policy=self.policy,
        )
        minimum = (
            len(warmup) * self.budget.encode_units
            + self.budget.materialize_units(len(warmup))
            + self.budget.base_future_units(len(warmup), stream_steps, query_count)
        )
        if minimum > self.max_units:
            raise ValueError("Budget is insufficient before episode start")
        self.history = []
        self.previous_action = Action.OFF
        self.previous_camera = None
        for observation in warmup:
            arrived = anchored_observation(observation, self.anchor_c2w)
            features = self.carrier.encode(arrived)
            self.budget.record("encode", self.budget.encode_units)
            self.cache = self.cache.append(arrived, features)
            self.previous_camera = arrived.camera
        self.scene_state = self.carrier.materialize(self.cache, self.fast)
        self.budget.record(
            "materialize",
            self.budget.materialize_units(self.cache.revision),
            views=self.cache.revision,
        )
        self.active = True
        self.sealed = False

    @torch.no_grad()
    def step(self, observation):
        if not self.active or self.sealed or self.remaining_steps == 0:
            raise ValueError("Episode is not accepting observations")
        if (
            observation.scene_id != self.scene_id
            or observation.camera.image_size != self.image_size
        ):
            raise ValueError("Observation scene/resolution differs from episode")
        if observation.frame_id <= self.cache.entries[-1].observation.frame_id:
            raise ValueError("Observations must arrive in strictly increasing frame order")
        if observation.rgb.is_cuda:
            torch.cuda.synchronize(observation.rgb.device)
        started = perf_counter()
        arrived = anchored_observation(observation, self.anchor_c2w)
        features = self.carrier.encode(arrived)
        self.budget.record("encode", self.budget.encode_units)
        # Current RGB is used only for comparison here; the cache and fast state are still old.
        feedback = preupdate_feedback(
            self.scene_state,
            arrived,
            self.renderer,
            image_feature=features,
            previous_camera=self.previous_camera,
        )
        self.budget.record("feedback_render", self.budget.render_units)
        control = ControlInput(
            **feedback,
            current_image_mean=float(arrived.rgb.mean()),
            density_mean=float(self.scene_state.density_logits.mean()),
            fast_norm=float(self.fast.delta_fuse.norm() + self.fast.delta_complete.norm()),
            observed_count=self.cache.revision,
            previous_action=self.previous_action,
            remaining_budget=self.budget.remaining,
            fast_state_stats=(
                float(self.fast.delta_fuse.norm()),
                float(self.fast.delta_complete.norm()),
                float(self.fast.delta_fuse.norm() + self.fast.delta_complete.norm()),
                float(
                    torch.maximum(
                        self.fast.delta_fuse.abs().max(), self.fast.delta_complete.abs().max()
                    )
                ),
            ),
            remaining_steps=self.remaining_steps,
        )
        self.budget.record("policy", self.budget.policy_units)
        feasible = self.budget.feasible_actions(
            self.cache.revision + 1, self.remaining_steps - 1, self.query_count
        )
        action = Action(self.policy.choose(control, feasible))
        if action not in feasible:
            raise ValueError("Policy selected an infeasible action")
        self.cache = self.cache.append(arrived, features)
        if action != Action.OFF:
            trace = self.carrier.trace(self.cache, self.fast)
            self.budget.record(
                "trace",
                self.budget.materialize_units(self.cache.revision),
                views=self.cache.revision,
            )
            proposal = self.write_rule.propose(trace)
            c = self.carrier.config
            self.budget.record("proposal_pair", 2 * c.token_count * c.hidden_dim * c.expansion_dim)
            self.fast = commit_write(
                self.fast, proposal, action, cache_revision=self.cache.revision
            )
        self.scene_state = self.carrier.materialize(self.cache, self.fast)
        self.budget.record(
            "materialize",
            self.budget.materialize_units(self.cache.revision),
            views=self.cache.revision,
        )
        self.remaining_steps -= 1
        self.previous_action = action
        self.previous_camera = arrived.camera
        if observation.rgb.is_cuda:
            torch.cuda.synchronize(observation.rgb.device)
        record = {
            "frame_id": observation.frame_id,
            "action": str(action),
            "feedback": feedback,
            "control_input": control.to_serializable(),
            "observed_count": self.cache.revision,
            "fast_step": self.fast.step,
            "seconds": perf_counter() - started,
            "fast_fuse_norm": float(self.fast.delta_fuse.norm()),
            "fast_complete_norm": float(self.fast.delta_complete.norm()),
            "budget": self.budget.report(),
        }
        self.history.append(record)
        return record

    def seal(self):
        if not self.active or self.remaining_steps:
            raise ValueError("Cannot seal before the declared stream is consumed")
        state = clone_scene_state(self.scene_state)
        self.sealed = True
        config = {
            "carrier": asdict(self.carrier.config),
            "write": asdict(self.write_rule.write_config),
            "render_samples": self.renderer.n_samples,
            "budget": self.max_units,
        }
        return SealedScene(
            self.episode_id,
            self.scene_id,
            self.split_id,
            self.query_vault_id,
            state,
            hash_scene_state(state),
            tuple(e.observation.frame_id for e in self.cache.entries),
            self.checkpoint_hash,
            hash_value(config),
            hash_fast_content(self.fast),
            self.anchor_c2w.detach().clone(),
        )
