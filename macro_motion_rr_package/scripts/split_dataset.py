import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--group_col", default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.metadata).copy()
    df["split"] = ""

    if args.group_col and args.group_col in df.columns:
        groups = df[args.group_col].astype(str).unique()
        train_g, test_g = train_test_split(groups, test_size=args.test_frac, random_state=args.seed)
        rel_val = args.val_frac / (1.0 - args.test_frac)
        train_g, val_g = train_test_split(train_g, test_size=rel_val, random_state=args.seed)
        df.loc[df[args.group_col].astype(str).isin(train_g), "split"] = "train"
        df.loc[df[args.group_col].astype(str).isin(val_g), "split"] = "val"
        df.loc[df[args.group_col].astype(str).isin(test_g), "split"] = "test"
    else:
        idx = df.index.values
        train_idx, test_idx = train_test_split(idx, test_size=args.test_frac, random_state=args.seed)
        rel_val = args.val_frac / (1.0 - args.test_frac)
        train_idx, val_idx = train_test_split(train_idx, test_size=rel_val, random_state=args.seed)
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    df.to_csv(args.out_csv, index=False)
    print(df["split"].value_counts())
    print(f"Wrote {args.out_csv}")

if __name__ == "__main__":
    main()
