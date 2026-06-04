# Remote Experiments

## Current State

- DICOM-backed projection pairs with angle separation `>= 25°` are prepared in [outputs_dicom_projection_pairs](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/outputs_dicom_projection_pairs).
- Reconstruction-ready cases are exported in [reconstruction_cases](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/reconstruction_cases).
- Candidate methods and priorities are listed in [reconstruction_candidates.json](/home/nosignalx2k/Рабочий стол/work/БСМП/3d/reconstruction_candidates.json).

## Remote Access

Current SSH endpoint:

```bash
ssh -p 26663 root@ssh7.vast.ai -L 8080:localhost:8080
```

Current blocker: at the time of preparation, `ssh7.vast.ai:26663` returned `Connection refused`.

## Local Preparation Steps

1. Refresh DICOM-backed projection pairs:

```bash
cd "/home/nosignalx2k/Рабочий стол/work/БСМП/3d"
./.venv-dicom/bin/python prepare_dicom_projection_pairs.py
```

2. Export reconstruction-ready cases:

```bash
./.venv-dicom/bin/python export_reconstruction_cases.py
```

## Recommended Remote Layout

```text
/root/coronary-recon/
  project/
  envs/
  repos/
  data/
  logs/
  results/
```

## Initial Sync Command

```bash
rsync -av --delete \
  --exclude '.venv-dicom' \
  --exclude '__pycache__' \
  "/home/nosignalx2k/Рабочий стол/work/БСМП/3d/" \
  root@ssh7.vast.ai:/root/coronary-recon/project/ \
  -e "ssh -p 26663"
```

## Practical Candidate Order

1. `neca`
2. `skeleton_triangulation_calibrated`
3. `graph_stereo_centerline`
4. `3dgr_car`
5. `sdf_car`

## Required Result Table

Each run should record:

- patient
- case_id
- series pair
- chosen phase pair
- method
- params
- reprojection metrics
- topology notes
- artifact paths
- status

## Minimum Artifact Set Per Run

- reconstructed volume / mesh / point cloud / centerline
- reprojection overlays for both views
- run config
- stdout/stderr log
- summary JSON row
