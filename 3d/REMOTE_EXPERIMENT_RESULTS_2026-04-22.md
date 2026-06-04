# Remote Experiment Results — 2026-04-22

## Remote environment
- Host: `50.35.188.60:36893`
- Workspace: `/workspace/coronary-recon`
- GPUs: `2 x NVIDIA RTX PRO 6000 Blackwell Server Edition`
- CUDA: `nvcc 13.0.88`
- Baseline env: `/venv/main`
- NeCA env: `/workspace/coronary-recon/venvs/neca-feasibility`

## Data prepared
- Candidate projection pairs with `angle_delta >= 25°`: [outputs_dicom_projection_pairs/projection_pairs_angle_ge25.json](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/outputs_dicom_projection_pairs/projection_pairs_angle_ge25.json)
- Reconstruction-ready case export: [reconstruction_cases](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/reconstruction_cases)
- NeCA case adapter: [prepare_neca_case.py](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/prepare_neca_case.py)
- NeCA evaluator: [evaluate_neca_case.py](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/evaluate_neca_case.py)

## Methods tested
### 1. Calibrated voxel baseline
- Script: [run_calibrated_voxel_baseline.py](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/run_calibrated_voxel_baseline.py)
- Batch runner: [run_baseline_experiments.py](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/run_baseline_experiments.py)
- Status: ran successfully on all 6 cases
- Local summaries:
  - [baseline_summary.json](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/baseline_summary.json)
  - [baseline_summary.csv](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/baseline_summary.csv)

Best overall baseline case:
- Case: `p0001__00000001__00000002`
- Phase index: `2`
- Variant: `soft_highres`
- Mean IoU: `0.427375`
- Mean Dice: `0.595444`
- Connected components: `5`
- Score total: `0.227375`
- Artifacts:
  - [results.json](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/baseline_best_p0001/results.json)
  - [overlay_a.png](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/baseline_best_p0001/overlay_a.png)
  - [overlay_b.png](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/baseline_best_p0001/overlay_b.png)

### 2. NeCA pilot
Repository:
- `https://github.com/WangStephen/NeCA`

What was required to make it run in the current container:
- Install modern stack in dedicated env: `torch 2.11 + odl + astra-toolbox + ninja + tensorboard`
- Compatibility patch in `hashencoder.cu`:
  - `inputs.type()` -> `inputs.scalar_type()`
  - `grad.type()` -> `grad.scalar_type()`
- Compatibility patch for ODL namespace changes:
  - `odl.tomo.*` -> `odl.applications.tomo.*`
  - `from odl.tomo.util.utility ...` -> `from odl.applications.tomo.util.utility ...`

Pilot runs executed on the same strongest baseline case `p0001__00000001__00000002`, phase `2`, detector `256`, volume `128^3`, `100` epochs.

#### NeCA with `mask`
- Output manifest: [manifest.json](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/neca_mask/manifest.json)
- Evaluation summary: [summary.json](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/neca_mask/summary.json)
- Overlays:
  - [overlay_a.png](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/neca_mask/overlay_a.png)
  - [overlay_b.png](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/neca_mask/overlay_b.png)
- Metrics:
  - Mean MSE: `0.221083`
  - Mean IoU: `0.055107`
  - Mean Dice: `0.104225`
  - Occupied voxels at threshold `0.5`: `0`
  - Components: `0`

#### NeCA with `masked_image`
- Output manifest: [manifest.json](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/neca_masked_image/manifest.json)
- Evaluation summary: [summary.json](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/neca_masked_image/summary.json)
- Overlays:
  - [overlay_a.png](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/remote_artifacts/20260422/neca_masked_image/overlay_a.png)
- Metrics:
  - Mean MSE: `0.203192`
  - Mean IoU: `0.016440`
  - Mean Dice: `0.032328`
  - Occupied voxels at threshold `0.5`: `0`
  - Components: `0`

## Current best solution
Current best practically working solution is the calibrated voxel baseline, not NeCA.

Why:
- It ran successfully across all 6 prepared cases without manual intervention.
- It produced topologically non-empty reconstructions with saved point clouds and reprojection overlays.
- Its best reprojection overlap is materially better than current NeCA adaptation on our real angiographic inputs.
- NeCA is now technically launchable, but under the current adaptation it over-smooths and does not yet produce a usable binary 3D vessel volume.

## What failed or remains incomplete
- `3DGR-CAR` was cloned but not yet adapted or benchmarked.
- `SDF-CAR` was not attempted yet.
- NeCA still needs a better input formulation and post-processing before it is competitive on our data.

## Most likely next improvements
1. Sweep NeCA over `detector_size`, `volume_extent_mm`, `bound`, and longer schedules such as `300-1000` epochs.
2. Try a softer target than raw binary mask, e.g. distance-transform projections or blurred vessel masks.
3. Threshold and topology post-processing for NeCA volumes before judging by binary overlap alone.
4. Add a second non-neural stereo baseline, e.g. graph/centerline stereo, for comparison against voxel carving.
