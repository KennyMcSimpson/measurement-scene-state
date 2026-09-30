# V11 launch incident (2026-09-29 18:18 remote time)

All six training subprocesses stopped at `destination.mkdir(exist_ok=False)` with FileNotFoundError: the parent `checkpoints/` directory had not been created at launch (earlier experiments created it in the launch command). No training step ran, no TRAIN, DEV or EVAL-V3 media was read and no metric was computed.

The six attempts are archived here by the runner's own `archive_attempt` (incident kind `launch_missing_checkpoints_directory`, recorded in `audit/incidents.json`) and count against the frozen infrastructure retry budget. The identical frozen commands are rerun after creating the directory. No frozen file, code or configuration changed.
