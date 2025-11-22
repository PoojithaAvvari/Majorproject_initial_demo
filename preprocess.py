# src/preprocess.py
import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit

def build_iu_metadata(reports_csv, projections_csv, images_dir, out_csv='data/metadata.csv'):
    # Load CSVs
    print("Loading:", reports_csv, projections_csv)
    reports = pd.read_csv(reports_csv)
    proj = pd.read_csv(projections_csv)

    print("reports columns:", list(reports.columns))
    print("projections columns:", list(proj.columns))

    # Ensure 'uid' exists in both
    if 'uid' not in reports.columns or 'uid' not in proj.columns:
        raise ValueError("Both CSVs must contain 'uid' column.")

    # Merge using UID
    df = proj.merge(reports, on="uid", how="left", suffixes=('_proj','_rep'))

    # Build image path from filename column
    if 'filename' not in df.columns:
        raise ValueError("'filename' column not found in projections CSV.")

    df["image_path"] = df["filename"].apply(lambda x: os.path.join(images_dir, str(x)))

    # Report number of missing files
    missing_imgs = df[~df["image_path"].apply(os.path.exists)]
    if len(missing_imgs) > 0:
        print(f"Warning: {len(missing_imgs)} image files referenced in CSV were NOT found under {images_dir}.")
        # show first few missing
        print(missing_imgs[["uid","filename"]].head())

    # Keep only useful columns (only if they exist)
    keep_cols = ["uid", "filename", "image_path", "projection"]
    # add findings/impression if present
    for c in ["findings", "impression", "image", "indication", "comparison", "MeSH", "Problems"]:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    df_out = df[keep_cols].copy()
    df_out.to_csv(out_csv, index=False)
    print("Saved metadata →", out_csv)
    print("Total records:", len(df_out))
    return df_out

def split_by_uid(in_csv="data/metadata.csv", out_prefix="data/"):
    df = pd.read_csv(in_csv)
    if 'uid' not in df.columns:
        raise ValueError("metadata.csv must contain 'uid' column to split by patient/study.")
    # group by uid to avoid leakage
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["uid"]))
    df.iloc[train_idx].to_csv(os.path.join(out_prefix, "train.csv"), index=False)
    df.iloc[test_idx].to_csv(os.path.join(out_prefix, "test.csv"), index=False)
    print("Train size:", len(train_idx))
    print("Test size:", len(test_idx))
    return os.path.join(out_prefix, "train.csv"), os.path.join(out_prefix, "test.csv")

if __name__ == "__main__":
    # Update these paths if your structure differs
    reports_csv = "archive/indiana_reports.csv"
    projections_csv = "archive/indiana_projections.csv"
    images_dir = "archive/images/images_normalized"
    out_csv = "data/metadata.csv"

    os.makedirs("data", exist_ok=True)

    df = build_iu_metadata(
        reports_csv=reports_csv,
        projections_csv=projections_csv,
        images_dir=images_dir,
        out_csv=out_csv
    )

    split_by_uid(out_csv)
