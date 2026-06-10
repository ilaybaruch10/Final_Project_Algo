import argparse, json
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from macro_motion.data.dataset import MacroMotionDataset
from macro_motion.models.macro_motion_model import MacroMotionModel
from macro_motion.metrics.rr_metrics import compute_regression_metrics
from macro_motion.eval.plots import plot_true_vs_pred, plot_time_series, plot_bland_altman

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

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {args.split}"):
            clip = batch["clip"].to(device)
            rr = batch["rr"].to(device)
            cond_id = batch["cond_id"].to(device)
            face_mask = batch.get("face_mask", None)
            if face_mask is not None: face_mask = face_mask.to(device)
            preds = model(clip, cond_id, face_mask=face_mask, unet_no_grad=True)
            for i in range(clip.size(0)):
                rows.append({"clip_path": batch["clip_path"][i], "rr_true": float(rr[i].cpu()),
                             "rr_pred": float(preds["rr_pred"][i].cpu()),
                             "error": float(preds["rr_pred"][i].cpu() - rr[i].cpu())})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"predictions_{args.split}.csv", index=False)
    metrics = compute_regression_metrics(df["rr_true"].values, df["rr_pred"].values)
    (out_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    plot_true_vs_pred(df["rr_true"].values, df["rr_pred"].values, out_dir / "true_vs_pred.png")
    plot_time_series(df["rr_true"].values, df["rr_pred"].values, out_dir / "rr_timeseries.png")
    plot_bland_altman(df["rr_true"].values, df["rr_pred"].values, out_dir / "bland_altman.png")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
