# NeCA Sweep Results (2026-04-22)

## Setup
- Remote GPU host: `50.35.188.60:36893`
- Repository: `/workspace/coronary-recon/repos/NeCA`
- Project workspace: `/workspace/coronary-recon/project`
- Cases: `p0001__00000001__00000002`, `p0001__00000001__00000003`, `p0001__00000002__00000003`, `p0006__00000001__00000002`
- Jobs: `24`
- Detector size: `256`
- Volume size: `128`
- Volume extent: `160 mm`
- Bound: `0.25`
- Epochs: `120`

## Sweep axes
- Input modes: `mask`, `edt_mask`, `blurred_mask`
- Temporal bundle sizes: `1`, `3`
- Phase index: baseline-best phase for each case

## Best NeCA result
- Case: `p0001__00000001__00000002`
- Mode: `edt_mask`
- Bundle: `3`
- Mean hard-mask IoU: `0.075047`
- Mean hard-mask Dice: `0.138848`
- Mean training-target IoU: `0.110885`
- Mean training-target Dice: `0.197968`
- Volume thresholding note: absolute `0.5` is not meaningful for NeCA outputs here; percentiles are more informative.

## Main findings
- `bundle=3` consistently beats `bundle=1` for all three target families.
- `edt_mask` is the best mode overall, but only slightly ahead of `mask` and `blurred_mask`.
- The gain is real but limited: best NeCA Dice `0.138848` is still far below the calibrated voxel baseline best Dice `0.595444`.
- NeCA volumes are low-amplitude and should be interpreted with relative thresholds, not a fixed `0.5` occupancy cutoff.

## Aggregated trend by mode
- `mask, bundle=1`: mean Dice `0.078351`
- `edt_mask, bundle=1`: mean Dice `0.078669`
- `blurred_mask, bundle=1`: mean Dice `0.078257`
- `mask, bundle=3`: mean Dice `0.121263`
- `edt_mask, bundle=3`: mean Dice `0.121408`
- `blurred_mask, bundle=3`: mean Dice `0.121217`

## Conclusion
- The best currently working open-source practical solution on these data remains the calibrated voxel baseline.
- The most promising NeCA direction is now clear: keep temporal bundles, keep soft targets, and next tune geometry/representation rather than going back to single-frame hard masks.
