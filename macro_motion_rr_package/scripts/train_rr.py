import argparse, json, yaml
from pathlib import Path
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from macro_motion.utils.seed import set_seed
from macro_motion.data.datamodule import make_loaders
from macro_motion.models.macro_motion_model import MacroMotionModel
from macro_motion.losses.losses import UncertaintyWeightedRRTempLoss
from macro_motion.metrics.rr_metrics import compute_regression_metrics

def str2bool(x):
    if isinstance(x, bool): return x
    return str(x).lower() in ["1", "true", "yes", "y"]

def downsample_rr_seq(rr_seq_raw, temporal_downsample):
    if temporal_downsample is None or temporal_downsample <= 1:
        return rr_seq_raw
    return rr_seq_raw[:, ::temporal_downsample]

@torch.no_grad()
def evaluate(model, loader, device, include_temp, criterion, cfg):
    model.eval()
    ys_frame, ps_frame = [], []
    total_loss, n = 0.0, 0
    ds = cfg["model"]["temporal_downsample"]

    for batch in loader:
        clip = batch["clip"].to(device)
        rr_seq = downsample_rr_seq(batch["rr_seq"].to(device), ds)
        cond_id = batch["cond_id"].to(device)
        face_mask = batch.get("face_mask", None)
        if face_mask is not None: face_mask = face_mask.to(device)

        preds = model(clip, cond_id, face_mask=face_mask, unet_no_grad=True)
        loss, _ = criterion(preds, rr_seq)

        total_loss += loss.item() * clip.size(0); n += clip.size(0)
        ys_frame.extend(rr_seq.detach().cpu().reshape(-1).numpy().tolist())
        ps_frame.extend(preds["rr_pred_seq"].detach().cpu().reshape(-1).numpy().tolist())

    metrics = compute_regression_metrics(ys_frame, ps_frame)
    return {f"frame_{k}": v for k, v in metrics.items()} | {"loss": total_loss / max(n,1)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--include_temp_path", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.include_temp_path is not None:
        cfg["model"]["include_temp_path"] = str2bool(args.include_temp_path)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    include_temp = bool(cfg["model"]["include_temp_path"])

    train_loader, val_loader, _ = make_loaders(
        args.metadata, cfg["data"]["input_size"], cfg["data"]["clip_frames"],
        cfg["train"]["batch_size"], cfg["train"]["num_workers"], include_temp
    )

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

    criterion = UncertaintyWeightedRRTempLoss(
        cfg["train"]["rr_loss"],
        cfg["train"]["huber_delta"],
        cfg["train"].get("rr_smooth_w", 0.05)
    ).to(device)

    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))
    best_mae = float("inf")
    ds = cfg["model"]["temporal_downsample"]

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            clip = batch["clip"].to(device)
            rr_seq = downsample_rr_seq(batch["rr_seq"].to(device), ds)
            cond_id = batch["cond_id"].to(device)
            face_mask = batch.get("face_mask", None)
            if face_mask is not None: face_mask = face_mask.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg["train"]["amp"])):
                preds = model(clip, cond_id, face_mask=face_mask, unet_no_grad=True)
                loss, terms = criterion(preds, rr_seq)

            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss.detach().cpu()), rr=float(terms["rr_loss"].cpu()))

        val = evaluate(model, val_loader, device, include_temp, criterion, cfg)
        print("val", val)
        with open(out_dir / "val_log.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, **val}) + "\\n")
        ckpt = {"model": model.state_dict(), "criterion": criterion.state_dict(), "cfg": cfg, "epoch": epoch, "val_metrics": val}
        if val["frame_MAE"] < best_mae:
            best_mae = val["frame_MAE"]
            torch.save(ckpt, out_dir / "best.pt")
        if epoch % cfg["train"]["save_every"] == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch}.pt")
    print("Best val frame MAE:", best_mae)

if __name__ == "__main__":
    main()
