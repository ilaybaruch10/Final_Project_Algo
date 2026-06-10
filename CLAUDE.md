# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

Final project: respiratory-rate (RR) estimation for NICU infants from NIR video. Two independent subprojects with no shared code and **incompatible Python environments** — keep separate venvs:

- `macro_motion_rr_package/` — the project's own PyTorch package. All new development happens here.
- `deep_motion_mag/` — third-party MIT "Learning-based Video Motion Magnification" (TensorFlow 1.x, Python 3.8 era). Used as an external tool to amplify breathing motion in videos; do not refactor it, only configure/run it.

No tests or linters are configured in either subproject.

## macro_motion_rr_package (PyTorch)

`macro_motion` is a plain package directory (no setup.py/pyproject), so run all scripts from inside `macro_motion_rr_package/`:

```bash
pip install -r requirements.txt

# Subject-grouped train/val/test split (adds a `split` column)
python scripts/split_dataset.py --metadata data/metadata.csv --out_csv data/metadata_split.csv --group_col subject_id

# Train RR branch (config defaults from configs/default.yaml, CLI can override include_temp_path)
python scripts/train_rr.py --metadata data/metadata_split.csv --out_dir runs/rr_baseline --include_temp_path false

# Evaluate a checkpoint
python scripts/evaluate_rr.py --metadata data/metadata_split.csv --ckpt runs/rr_baseline/best.pt --split test --out_dir runs/rr_baseline/eval_test

# Pre-train the NIR->IR U-Net separately (only needed when include_temp_path=true)
python scripts/train_unet_nir2ir.py --pairs_csv data/nir_ir_pairs.csv --out_dir runs/unet
```

### Architecture

Input: 20 s NIR clip, 600 frames @ 30 Hz, shapes `(T,H,W,C)` or `(T,C,H,W)`, loaded from `.npy`/`.npz`/frame folders.

RR path (`MacroMotionModel.forward` in `macro_motion/models/macro_motion_model.py`):
normalize to [0,1] → temporal downsample ×3 (→ 200 frames @ 10 Hz) → grayscale (channel mean) → `ResNetFrameEncoder1Ch` (ResNet-50 with 1-channel conv1, per-frame) → Linear 2048→d_model → `SharedMacroTransformer` → RR head (scalar).

Optional thermal path (`include_temp_path`, off by default): same clip → `UNetNIR2IR` (frozen during RR training via `unet_no_grad=True`) → `FaceAwareResNetEncoder1Ch` with face mask downsampled to 7×7 → the **same shared transformer** → temp head.

`SharedMacroTransformer` (`macro_motion/models/transformer.py`) prepends 3 tokens to the frame sequence: `[CLS]`, a domain embedding (`DOMAIN_RR=0` / `DOMAIN_TEMP=1`), and a condition embedding (`COND_EXPOSED=0` / `COND_BLANKET=1`); the `[CLS]` output feeds the heads. Both paths share this transformer — that's the multi-task design.

Loss (`macro_motion/losses/losses.py`): `UncertaintyWeightedRRTempLoss`, Kendall-style learnable log-sigma weighting between RR (Huber by default) and temp (MSE, masked by `temp_valid` and weighted per condition). **The criterion has trainable parameters** — `train_rr.py` adds `criterion.parameters()` to the optimizer and saves its state_dict in checkpoints; preserve this when touching the training loop.

Component map:
- `macro_motion/data/dataset.py` — `MacroMotionDataset`, reads metadata CSV, pads/truncates clips to `clip_frames`, resizes to `input_size`
- `macro_motion/data/datamodule.py` — `make_loaders` (train/val/test)
- `macro_motion/utils/io.py` — `load_clip` / `load_mask` / `normalize_to_01`
- `macro_motion/metrics/rr_metrics.py` — MAE, RMSE, MAPE, bias, SD error, Pearson r, R²
- `macro_motion/eval/plots.py` — evaluation plots

### Dataset CSV

Required columns: `clip_path,rr_bpm,subject_id` (plus `split` after running split_dataset.py). Optional: `temp_c`, `face_mask_path`, `cond` (`exposed`/`blanket`; missing → exposed). Missing `temp_c` rows get `temp_valid=0` so they don't contribute to the temp loss.

## deep_motion_mag (TensorFlow 1.x)

Run via **Git Bash** on this Windows machine (see `video magnification commands.txt`). Requires the pretrained checkpoint under `data/training/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3/checkpoint` and input frames as PNGs under `data/input/<video_name>`:

```bash
cd deep_motion_mag
# Static mode (first frame as reference) / add "yes" for dynamic (consecutive-frame) mode
sh run_on_test_videos.sh o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3 <video_name> <amplification_factor> [yes]
# Temporal band-pass filter mode
sh run_temporal_on_test_videos.sh o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3 <video_name> <amp> <low_cutoff> <high_cutoff> <fps> <n_taps> <filter_type>
```

Outputs land under `data/output/`. `frames2vid.py` assembles output frames into a video.
