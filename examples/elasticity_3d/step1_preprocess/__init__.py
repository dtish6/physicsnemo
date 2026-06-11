# Step 1 — Preprocessing.
#
# Turns raw apartment floor-plan CSVs into training-ready HDF5 samples. The
# active flow is 1:1 (no downsampling, no padding): each plan is binarised
# straight into a fixed voxel grid and cropped to fit, then step2 solves it.
#
#   _1_1_binarize        rasterize CSV -> fixed (D,H,W) binary grid, crop to box
#   generate_from_csv    driver: binarize -> step2 FEA solve -> write sample_*.h5
#
# Run as a script, e.g.:
#     python step1_preprocess/generate_from_csv.py --jobs 3 --precond direct
#
# support/ holds non-pipeline helpers:
#   check_bbox.py        report which plans exceed the box (crop diagnostics)
#   generate_mock_hdf5.py  synthetic samples for smoke-testing
#   _1_2_downsample.py / _1_3_pad.py / _1_4_pad_to_cube.py
#                        retired: used by the older downsample+pad-to-cube flow,
#                        kept for reference; not part of the 1:1 direct pipeline.
