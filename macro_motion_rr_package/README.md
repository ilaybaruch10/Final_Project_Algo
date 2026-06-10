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
   -> keep 3-channel NIR/RGB input
   -> frozen built-in PyTorch pretrained ResNet-50 feature extractor
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


## Frozen pretrained ResNet-50 update

This package version keeps PyTorch's built-in pretrained ResNet-50 unchanged and frozen:
- `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`
- original `conv1` stays 3-channel
- `fc = Identity()`
- `eval()`
- all ResNet parameters have `requires_grad=False`

For RR:
- NIR/RGB 3-ch -> frozen ResNet-50 unchanged -> Linear(2048->d_model)

For Temp:
- IRhat 1-ch -> repeat to 3-ch -> frozen ResNet-50 trunk -> face-aware attention pooling -> Linear(2048->d_model)


## RR path correction

The RR path now keeps the original 3-channel NIR/RGB clip:
```text
NIR/RGB 3-ch clip -> frozen pretrained ResNet-50 -> Linear(2048->512) -> Transformer -> RR
```

No grayscale conversion and no 1-channel-to-3-channel replication are used in the active RR path.


## DOMAIN token flag

The package now has:

```yaml
model:
  use_domain_token: false
```

For current RR-only training, keep it `false`.

RR-only Transformer input becomes:

```text
[CLS] [COND] z1 z2 ... zT
```

With 600 raw frames and `temporal_downsample=3`, `T=200`, so the Transformer input is:

```text
(B, 202, 512)
```

If you later enable the Temp path and want a shared RR+Temp Transformer, set:

```yaml
model:
  include_temp_path: true
  use_domain_token: true
```

Then the sequence becomes:

```text
[CLS] [DOMAIN] [COND] z1 z2 ... zT
```

with shape `(B, T+3, 512)`.


## Dense per-frame RR output update

The model now supports dense RR prediction.

Current RR-only output:
```text
Input:              (B, 600, 3, 512, 512)
After downsample:   (B, 200, 3, 512, 512)
Frame tokens:       (B, 200, 512)
Transformer output: (B, 202, 512)   # [CLS][COND] + 200 frame tokens
RR output sequence: (B, 200)
RR scalar output:   mean(RR sequence) -> (B,)
```

Use dense labels with:

```csv
clip_path,rr_path,subject_id,cond
data/clips/clip_001.npy,data/rr/clip_001_rr.npy,baby_001,exposed
```

`rr_path` may be `.npy`, `.npz`, `.csv`, or `.txt`.

Alternative:

```csv
clip_path,rr_seq,subject_id,cond
data/clips/clip_001.npy,"42;42;43;43;...",baby_001,exposed
```

Backward compatibility:

```csv
clip_path,rr_bpm,subject_id,cond
data/clips/clip_001.npy,42.0,baby_001,exposed
```

In scalar mode, RR is repeated across all frames.

Training loss:
```text
Huber(rr_pred_seq, rr_true_seq) + rr_smooth_w * |RR_t - RR_{t-1}|
```

Evaluation outputs:
```text
predictions_framewise_test.csv
predictions_clipmean_test.csv
metrics_test.json
```
