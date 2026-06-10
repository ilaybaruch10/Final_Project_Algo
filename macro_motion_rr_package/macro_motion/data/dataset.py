import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from macro_motion.utils.io import load_clip, normalize_to_01, load_mask

COND_TO_ID = {"exposed": 0, "blanket": 1}

class MacroMotionDataset(Dataset):
    """
    Required CSV columns: clip_path, rr_bpm, split.
    Optional: temp_c, face_mask_path, cond.
    """
    def __init__(self, metadata_csv, split="train", input_size=224,
                 clip_frames=600, normalize=True, include_temp=False):
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split={split}")
        self.input_size = input_size
        self.clip_frames = clip_frames
        self.normalize = normalize
        self.include_temp = include_temp

    def __len__(self):
        return len(self.df)

    def _fix_length(self, x):
        T = x.shape[0]
        if T == self.clip_frames:
            return x
        if T > self.clip_frames:
            return x[:self.clip_frames]
        pad = self.clip_frames - T
        return torch.cat([x, x[-1:].repeat(pad, 1, 1, 1)], dim=0)

    def _resize(self, x, mode="bilinear"):
        if mode == "nearest":
            return F.interpolate(x, size=(self.input_size, self.input_size), mode=mode)
        return F.interpolate(x, size=(self.input_size, self.input_size), mode=mode, align_corners=False)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip = load_clip(row["clip_path"])
        if clip.shape[1] < 3:
            clip = clip.repeat(1, 3, 1, 1)
        clip = clip[:, :3]
        if self.normalize:
            clip = normalize_to_01(clip)
        clip = self._fix_length(clip)
        clip = self._resize(clip)

        rr = torch.tensor(float(row["rr_bpm"]), dtype=torch.float32)
        cond = str(row["cond"]).lower() if "cond" in row and pd.notna(row["cond"]) else "exposed"
        cond_id = torch.tensor(COND_TO_ID.get(cond, 0), dtype=torch.long)

        sample = {
            "clip": clip,
            "rr": rr,
            "cond_id": cond_id,
            "clip_path": str(row["clip_path"]),
        }

        if self.include_temp:
            has_temp = "temp_c" in row and pd.notna(row["temp_c"])
            sample["temp"] = torch.tensor(float(row["temp_c"]) if has_temp else 0.0, dtype=torch.float32)
            sample["temp_valid"] = torch.tensor(1.0 if has_temp else 0.0, dtype=torch.float32)

            mask = None
            if "face_mask_path" in row and pd.notna(row["face_mask_path"]):
                mask = load_mask(row["face_mask_path"])
            if mask is None:
                mask = torch.ones((self.clip_frames, 1, self.input_size, self.input_size), dtype=torch.float32)
            else:
                mask = self._fix_length(mask)
                mask = self._resize(mask, mode="nearest")
            sample["face_mask"] = mask

        return sample
