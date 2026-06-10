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

@torch.no_grad()
def evaluate(model, loader, device, include_temp, criterion, w_blanket, w_exposed):
    model.eval()
    ys, ps = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        clip = batch["clip"].to(device)
        rr = batch["rr"].to(device)
        cond_id = batch["cond_id"].to(device)
        face_mask = batch.get("face_mask", None)
        if face_mask is not None: face_mask = face_mask.to(device)
        preds = model(clip, cond_id, face_mask=face_mask, unet_no_grad=True)
        if include_temp:
            temp = batch["temp"].to(device)
            temp_valid = batch["temp_valid"].to(device)
            w_temp = torch.where(cond_id == 1, torch.full_like(temp, w_blanket), torch.full_like(temp, w_exposed))
            loss, _ = criterion(preds, rr, temp=temp, temp_valid=temp_valid, w_temp=w_temp)
        else:
            loss, _ = criterion(preds, rr)
        total_loss += loss.item() * clip.size(0); n += clip.size(0)
        ys.extend(rr.detach().cpu().numpy().tolist())
        ps.extend(preds["rr_pred"].detach().cpu().numpy().tolist())
    m = compute_regression_metrics(ys, ps); m["loss"] = total_loss / max(n,1)
    return m

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
        include_temp_path=include_temp, unet_base=cfg["model"]["unet_base"]
    ).to(device)

    criterion = UncertaintyWeightedRRTempLoss(cfg["train"]["rr_loss"], cfg["train"]["huber_delta"]).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))
    best_mae = float("inf")

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            clip = batch["clip"].to(device)
            rr = batch["rr"].to(device)
            cond_id = batch["cond_id"].to(device)
            face_mask = batch.get("face_mask", None)
            if face_mask is not None: face_mask = face_mask.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg["train"]["amp"])):
                preds = model(clip, cond_id, face_mask=face_mask, unet_no_grad=True)
                if include_temp:
                    temp = batch["temp"].to(device)
                    temp_valid = batch["temp_valid"].to(device)
                    w_temp = torch.where(cond_id == 1,
                                         torch.full_like(temp, cfg["train"]["temp_loss_weight_blanket"]),
                                         torch.full_like(temp, cfg["train"]["temp_loss_weight_exposed"]))
                    loss, _ = criterion(preds, rr, temp=temp, temp_valid=temp_valid, w_temp=w_temp)
                else:
                    loss, _ = criterion(preds, rr)
            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        val = evaluate(model, val_loader, device, include_temp, criterion,
                       cfg["train"]["temp_loss_weight_blanket"], cfg["train"]["temp_loss_weight_exposed"])
        print("val", val)
        with open(out_dir / "val_log.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, **val}) + "\n")
        ckpt = {"model": model.state_dict(), "criterion": criterion.state_dict(), "cfg": cfg, "epoch": epoch, "val_metrics": val}
        if val["MAE"] < best_mae:
            best_mae = val["MAE"]
            torch.save(ckpt, out_dir / "best.pt")
        if epoch % cfg["train"]["save_every"] == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch}.pt")
    print("Best val MAE:", best_mae)

if __name__ == "__main__":
    main()
