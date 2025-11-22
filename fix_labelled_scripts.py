# src/fix_labeled_splits.py
import pandas as pd

for path in ["data/train_labeled.csv", "data/test_labeled.csv"]:
    print("Processing", path)
    df = pd.read_csv(path)

    # choose the correct image_path column (prefer _y from metadata)
    if "image_path" in df.columns:
        print("  already has image_path")
    else:
        if "image_path_y" in df.columns:
            df["image_path"] = df["image_path_y"]
            print("  used image_path_y")
        elif "image_path_x" in df.columns:
            df["image_path"] = df["image_path_x"]
            print("  used image_path_x")
        else:
            raise RuntimeError("No image_path columns found in " + path)

    # Reorder columns: keep original order but ensure image_path, labels_vec, uncertain_vec exist
    cols = list(df.columns)
    # ensure labels_vec exists
    if "labels_vec" not in cols:
        raise RuntimeError(f"{path} missing labels_vec")
    # make a clean column order: put image_path and labels first for readability
    desired = ["uid","filename","image_path","projection","findings","impression","labels_vec","uncertain_vec","labels_readable"]
    # keep only columns that exist and append the rest
    new_cols = [c for c in desired if c in cols] + [c for c in cols if c not in desired]
    df = df[new_cols]

    df.to_csv(path, index=False)
    print("  saved cleaned", path)
