# Macro-Motion RR Package

PyTorch package for the NICU macro-motion RR branch.

## Main idea

Input clip:
- NIR video clip: `(T, C, H, W)` or `(T, H, W, C)`
- Default raw clip length: **20 s @ 30 Hz = 600 frames**
- Model can downsample internally to 10 Hz for RR, giving `T_model=200`

RR path:
```text
NIR 3-ch clip
   -> normalize to [0,1]
   -> optional temporal downsample 30 Hz -> 10 Hz
   -> grayscale 1-ch
   -> ResNet-50 with 1-channel conv1
   -> Linear 2048 -> d_model
   -> shared Transformer with [CLS], [RR_DOMAIN], [COND]
   -> RR head
   -> estimated RR scalar
```

Optional thermal path, controlled by `include_temp_path`:
```text
NIR 3-ch clip
   -> U-Net NIR->IRhat
   -> face-aware ResNet-50
   -> Linear 2048 -> d_model
   -> same shared Transformer with [CLS], [TEMP_DOMAIN], [COND]
   -> Temp head
```

## Dataset CSV format

Required columns:

```csv
clip_path,rr_bpm,subject_id
/path/to/clip_001.npy,42.0,baby_001
```

Optional columns:

```csv
temp_c,face_mask_path,cond
```

- `cond`: `exposed` or `blanket`
- `clip_path` may be `.npy`, `.npz`, or a folder of frame images.
- Supported clip shapes: `(T,H,W,C)` or `(T,C,H,W)`.

## Usage

Split:
```bash
python scripts/split_dataset.py --metadata data/metadata.csv --out_csv data/metadata_split.csv --group_col subject_id
```

Train RR only:
```bash
python scripts/train_rr.py --metadata data/metadata_split.csv --out_dir runs/rr_baseline --include_temp_path false
```

Evaluate:
```bash
python scripts/evaluate_rr.py --metadata data/metadata_split.csv --ckpt runs/rr_baseline/best.pt --split test --out_dir runs/rr_baseline/eval_test
```

## Defaults

- raw clip: 600 frames at 30 Hz
- model downsample: factor 3 -> 200 frames at 10 Hz
- d_model: 512
- n_heads: 8
- depth: 6
- dropout: 0.1
- RR loss: Huber
- metrics: MAE, RMSE, MAPE, bias, SD error, Pearson r, R2
