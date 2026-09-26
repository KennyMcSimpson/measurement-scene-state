# V2 development and run record

V1 artifact and source snapshot captured before V2 additions; no V1 result overwritten.
The supplied root protocol was absent and was copied byte-for-byte from the user attachment.
All new science modules are separate v2_* files. Original V1 numerical code is reused.

Source-annotated deployment is explicitly part of the existing task. New feature creation
accepts only source/target features, source mask, and distinct frame identifiers; it has
no target reward/mask argument. Discovery reward labels are reused from sealed V1 raw.

Online resource audit was authorized by the user. VOST metadata were range-downloaded;
clip IDs were not accepted as original video IDs. FBMS-59 metadata and official evaluation
code were checked; uncertain original video mapping and incompatible label interpretation
blocked selection. No external target GT was opened and no cohort was substituted silently.
The config retains the initially considered VOST domain for traceability; no external domain
was actually evaluated. This field is not evidence that VOST passed the audit.

Pre-confirmation review found and fixed a fail-open case for absent near-duplicate evidence:
confirmation now requires explicit original-video and near-duplicate verification. Fixture
coverage confirms absent evidence cannot pass. No independent labels were involved.

Routine lint failures during new-file development were corrected before full validation.
Prior V1 native crashes remain recorded in the unchanged V1 postprocessing incident files;
V2 numerical jobs use PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.
This setting is a reproducibility precaution, not a diagnosis of the old crashes.

V2 full-suite first attempt also exited 139 in legacy dynamic/types tensor validation.
Same command/environment retry passed 500 tests, with one preexisting checkpoint skip.
Both full logs and test_incident.json are preserved; root cause remains undetermined.
No legacy crashing code was changed to obtain the retry pass.
