import argparse, json
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from macro_motion.data.dataset import MacroMotionDataset
from macro_motion.models.macro_motion_model import MacroMotionModel
from macro_motion.metrics.rr_metrics import compute_regression_metrics
from macro_motion.eval.plots import plot_true_vs_pred, plot_time_series, plot_bland_altman

def downsample_rr_seq(rr_seq_raw, temporal_downsample):
    if temporal_downsample is None or temporal_downsample <= 1:
        return rr_seq_raw
    return rr_seq_raw[:, ::temporal_downsample]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    include_temp = bool(cfg["model"]["include_temp_path"])
    ds_factor = cfg["model"]["temporal_downsample"]

    ds = MacroMotionDataset(args.metadata, args.split, cfg["data"]["input_size"],
                            cfg["data"]["clip_frames"], True, include_temp)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MacroMotionModel(
        d_model=cfg["model"]["d_model"], n_heads=cfg["model"]["n_heads"],
        depth=cfg["model"]["depth"], dropout=cfg["model"]["dropout"],
        ff_mult=cfg["model"]["ff_mult"], max_len=cfg["model"]["max_len"],
        temporal_downsample=cfg["model"]["temporal_downsample"],
        pretrained_resnet=cfg["model"]["use_pretrained_resnet"],
        freeze_resnet_until=cfg["model"]["freeze_resnet_until"],
        include_temp_path=include_temp, unet_base=cfg["model"]["unet_base"],
        use_domain_token=cfg["model"].get("use_domain_token", False)
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    frame_rows, clip_rows = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {args.split}"):
            clip = batch["clip"].to(device)
            rr_seq = downsample_rr_seq(batch["rr_seq"].to(device), ds_factor)
            cond_id = batch["cond_id"].to(device)
            face_mask = batch.get("face_mask", None)
            if face_mask is not None: face_mask = face_mask.to(device)

            preds = model(clip, cond_id, face_mask=face_mask, unet_no_grad=True)
            pred_seq = preds["rr_pred_seq"]
            pred_clip = preds["rr_pred"]
            true_clip = rr_seq.mean(dim=1)

            for i in range(clip.size(0)):
                clip_rows.append({
                    "clip_path": batch["clip_path"][i],
                    "rr_true_mean": float(true_clip[i].cpu()),
                    "rr_pred_mean": float(pred_clip[i].cpu()),
                    "error_mean": float(pred_clip[i].cpu() - true_clip[i].cpu()),
                })
                for t in range(rr_seq.shape[1]):
                    frame_rows.append({
                        "clip_path": batch["clip_path"][i],
                        "t_index_model": t,
                        "rr_true": float(rr_seq[i, t].cpu()),
                        "rr_pred": float(pred_seq[i, t].cpu()),
                        "error": float(pred_seq[i, t].cpu() - rr_seq[i, t].cpu()),
                    })

    df_frame = pd.DataFrame(frame_rows)
    df_clip = pd.DataFrame(clip_rows)
    df_frame.to_csv(out_dir / f"predictions_framewise_{args.split}.csv", index=False)
    df_clip.to_csv(out_dir / f"predictions_clipmean_{args.split}.csv", index=False)

    frame_metrics = compute_regression_metrics(df_frame["rr_true"].values, df_frame["rr_pred"].values)
    clip_metrics = compute_regression_metrics(df_clip["rr_true_mean"].values, df_clip["rr_pred_mean"].values)
    metrics = {f"frame_{k}": v for k, v in frame_metrics.items()}
    metrics.update({f"clipmean_{k}": v for k, v in clip_metrics.items()})
    (out_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))

    plot_true_vs_pred(df_frame["rr_true"].values, df_frame["rr_pred"].values, out_dir / "true_vs_pred_framewise.png")
    plot_time_series(df_frame["rr_true"].values, df_frame["rr_pred"].values, out_dir / "rr_timeseries_framewise.png")
    plot_bland_altman(df_frame["rr_true"].values, df_frame["rr_pred"].values, out_dir / "bland_altman_framewise.png")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
