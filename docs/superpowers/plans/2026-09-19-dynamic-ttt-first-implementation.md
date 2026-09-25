# Dynamic TTT first executable implementation

Authorization: user requested extraction and starting modular implementation/debugging on 2026-09-19. Source design: `../specs/2026-09-18-dynamic-ttt-modular-code-design.md`.

This first delivery makes the development episode executable: strict online/query readers; independent observation cache and fast weights; a differentiable typed carrier; direct writes and causal runner; isolated post-seal evaluation; real train/dev smoke and a small gradient/optimizer check. Trained teacher/policy and full benchmark training remain subsequent stages, not claimed by an untrained pilot.

Delivery record: `../../dynamic_ttt_implementation_20260919.md`. Download monitoring is paused; all 26 archives are extracted and independently checked for exact file paths and sizes. Final implementation checks are recorded under `outputs/dynamic_ttt_pilot_20260919/final_verification.json`.

1. Confirm download heartbeat is paused; extract all 26 verified archives to an isolated final holdout directory with path/CRC/space checks and receipts. Preserve ZIPs.
2. Add runtime types/config/cache and behavior tests; compile existing six-scene pilot into isolated online and query manifests.
3. Implement small carrier/lifting and two explicit writable matrices; reuse existing camera and fixed renderer contracts.
4. Implement direct write math, typed proposals, action policies, budget/call ledger, causal runner and sealed snapshots.
5. Add pilot CLI and limited offline gradient smoke. Run the four fixed action controls from identical initialization on train/dev only; report technical results, not scientific improvement.
6. Review boundaries independently, fix issues, run targeted and V5 regression tests, verify frozen baseline hashes, summarize implementation and extraction evidence.

Ownership: extraction agent owns extraction script/tests/output; carrier agent owns dynamic/carrier.py and lifting.py plus their test; data agent owns data/episodes.py and evaluation/sealed_queries.py plus compiler/tests. Root owns shared types/config/cache, write rule/runner/policy/feedback/budget/checkpoint, integration and reports. Existing static implementation and V5 files are not modified. No project-memory write is requested.

Validation priorities: future/query data cannot affect prefix state/actions; OFF appends observations; ALL is synchronous; unselected matrices and slow parameters remain unchanged during deployment; resets isolate scenes; finite gradients reach offline write parameters; seal does not alias live tensors; query evaluation cannot mutate state; actual calls and full-prefix rebuild costs are recorded.
