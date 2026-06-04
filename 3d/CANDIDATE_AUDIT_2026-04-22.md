# Candidate Audit (2026-04-22)

## Real-mask rerun
A strict rerun on the selected real phase-pair masks was completed.

Case:
- `p0001__00000001__00000002`
- phase pair: `f0032 / f0039`
- no `temporal_refined`
- no `fused`

Artifacts:
- `outputs_p0001_realmask_phasepair/p0001__00000001__00000002_realmask_summary.json`
- `outputs_p0001_realmask_phasepair/p0001__00000001__00000002_realmask_tube_graph.obj`
- `outputs_p0001_realmask_phasepair/p0001__00000001__00000002_realmask_overlay_a.png`
- `outputs_p0001_realmask_phasepair/p0001__00000001__00000002_realmask_overlay_b.png`

Result:
- matching degrades noticeably on real masks only
- `proximal_segments = 0`
- `leaf_pairs = 2`
- `correspondences = 18`
- `inliers = 9`
- `inlier_ratio = 0.5`
- `median_sampson = 22.0609`

Interpretation:
- real masks are the right source for final geometry
- but current correspondence logic is not strong enough to stabilize on them without temporal scaffolding

## 3DGR-CAR
Remote path:
- `/workspace/coronary-recon/repos/3DGR-CAR`

Status:
- cloned
- not adapted to our data yet
- not benchmarked yet

Observed repository state:
- code lives under `3dgs-car/`
- README asks for old stack: `torch==2.0.0`, CUDA 11.7, manual ODL install, ASTRA, Gaussian Splatting submodules
- `requirements.txt` is extremely large and noisy, with many unrelated packages
- current remote environment is not aligned to this stack

Practical conclusion:
- `3DGR-CAR` is available, but bringing it up cleanly will require environment isolation and input adapter work
- this is a medium/high-effort candidate, not a quick benchmark

## SDF-CAR
Remote path:
- `/workspace/coronary-recon/repos/SDF-CAR`

Status:
- cloned
- audited
- not yet adapted to our data
- not benchmarked on our cases

Critical code finding:
- current implementation is not directly patient-specific from two input masks
- `src/trainer.py` explicitly loads a 3D ground-truth volume from `input_data_dir/{model_id}.npy`
- it then generates the two projections from that 3D volume internally
- it also generates 2D SDF targets from those GT-based projections

Implication:
- in its current form, this repo is not a drop-in benchmark for our setting
- it still assumes access to 3D GT volumes during training setup
- using it on our angiographic pairs requires a real code adaptation, not just config changes

Practical conclusion:
- `SDF-CAR` looks conceptually closer to what we want than `3DGR-CAR`
- but the cloned codebase is still not immediately usable on our data
- the first adaptation step would be to replace GT-volume loading with direct ingestion of our paired projections/masks and externally supplied geometry

## Recommended next step
1. Stop treating `3DGR-CAR` and `SDF-CAR` as ready-made baselines.
2. Keep the real-mask rerun as the honest reference point.
3. If continuing with external methods, adapt `SDF-CAR` first, because its representation is more relevant and its codebase is cleaner.
4. In parallel, improve real-mask correspondence locally, because that remains the true bottleneck regardless of backbone.
