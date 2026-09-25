"""Run the frozen V8 Q1 label-blind supplier-risk qualification audit."""

from mcss.v8_q1_runner import (
    _check_drop_anchor_contract,
    _decide,
    _load_frozen_windows,
    _load_q0_gate,
    _quiet_joint_failure_masks,
    _select_request,
    _selective_severe_comparison_passes,
    main,
)

__all__ = [
    "_check_drop_anchor_contract",
    "_decide",
    "_load_frozen_windows",
    "_load_q0_gate",
    "_quiet_joint_failure_masks",
    "_selective_severe_comparison_passes",
    "_select_request",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
