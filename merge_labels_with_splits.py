# src/merge_labels_with_splits.py
import pandas as pd
import os

def normalize_fname_col(df, col):
    """Create a normalized filename column for robust matching."""
    return df[col].astype(str).str.strip().str.lower().apply(lambda x: os.path.basename(x))

def merge_on_uid_or_filename(meta_path="data/metadata_labeled.csv",
                             train_path="data/train.csv",
                             test_path="data/test.csv",
                             out_train="data/train_labeled.csv",
                             out_test="data/test_labeled.csv"):
    meta = pd.read_csv(meta_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Ensure expected label columns exist in meta
    if 'labels_vec' not in meta.columns:
        raise RuntimeError(f"{meta_path} does not contain 'labels_vec' column - ensure labeler.py was run.")

    # Try merging on uid if present in all
    if 'uid' in meta.columns and 'uid' in train.columns and 'uid' in test.columns:
        print("Merging on 'uid' ...")
        train_l = train.merge(meta[['uid','image_path','labels_vec','uncertain_vec','labels_readable']],
                              on='uid', how='left')
        test_l = test.merge(meta[['uid','image_path','labels_vec','uncertain_vec','labels_readable']],
                              on='uid', how='left')
    else:
        # Fallback: merge on normalized filename
        print("Merging on filename (normalized) ...")
        meta['__fname_n'] = normalize_fname_col(meta, 'filename' if 'filename' in meta.columns else 'image_path')
        train['__fname_n'] = normalize_fname_col(train, 'filename' if 'filename' in train.columns else 'image_path')
        test['__fname_n']  = normalize_fname_col(test,  'filename' if 'filename' in test.columns  else 'image_path')

        train_l = train.merge(meta[['__fname_n','image_path','labels_vec','uncertain_vec','labels_readable']],
                              left_on='__fname_n', right_on='__fname_n', how='left')
        test_l = test.merge(meta[['__fname_n','image_path','labels_vec','uncertain_vec','labels_readable']],
                              left_on='__fname_n', right_on='__fname_n', how='left')

        # drop helper cols
        meta.drop(columns=['__fname_n'], inplace=True, errors=False)

    # Save labeled splits
    os.makedirs(os.path.dirname(out_train) or ".", exist_ok=True)
    train_l.to_csv(out_train, index=False)
    test_l.to_csv(out_test, index=False)

    # Report missing labels
    n_missing_train = train_l['labels_vec'].isna().sum()
    n_missing_test  = test_l['labels_vec'].isna().sum()
    print(f"Saved {out_train} ({len(train_l)} rows) and {out_test} ({len(test_l)} rows).")
    print(f"Train rows with missing labels: {n_missing_train}")
    print(f"Test  rows with missing labels: {n_missing_test}")

    if n_missing_train > 0:
        print("\nFirst 20 train rows missing labels:")
        print(train_l[train_l['labels_vec'].isna()][['uid','filename','image_path']].head(20).to_string(index=False))

    if n_missing_test > 0:
        print("\nFirst 20 test rows missing labels:")
        print(test_l[test_l['labels_vec'].isna()][['uid','filename','image_path']].head(20).to_string(index=False))

    return out_train, out_test

if __name__ == "__main__":
    merge_on_uid_or_filename()
