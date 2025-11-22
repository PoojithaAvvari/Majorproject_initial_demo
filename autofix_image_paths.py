# src/autofix_image_paths.py
"""
Auto-fix image_path entries in data/train_labeled.csv and data/test_labeled.csv.

For any row where the CSV's image_path does not exist on disk, this script:
  - extracts the basename (filename) from the CSV path
  - searches under the `search_root` directory for files containing that basename
  - if found, replaces the CSV image_path with the actual found path
  - reports replacements and optionally saves changes (dry-run mode)

Usage:
  # dry-run (shows what WOULD change)
  python src/autofix_image_paths.py --dry_run

  # actually apply changes (backup created)
  python src/autofix_image_paths.py --apply
"""

import argparse, os, pandas as pd, sys
from pathlib import Path

SEARCH_ROOT = "archive/images"   # change if your images live somewhere else
TARGET_CSVS = ["data/train_labeled.csv", "data/test_labeled.csv"]
BACKUP_SUFFIX = ".bak_autofix"

def find_file_for_basename(basename, search_root):
    """
    Search recursively under search_root for any filename that contains basename.
    Returns the first match (full path) or None.
    """
    # normalize basename
    b = basename.strip().lower()
    for root, dirs, files in os.walk(search_root):
        for fn in files:
            if b in fn.lower():
                return os.path.join(root, fn)
    return None

def process_csv(path, search_root, dry_run=True):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] CSV not found: {path}")
        return 0, []

    df = pd.read_csv(path)
    replaced = []
    missing_before = 0
    for i, row in df.iterrows():
        # determine current path candidates
        cur = None
        # prefer explicit image_path if present
        for col in ("image_path", "image_path_y", "image_path_x"):
            if col in df.columns:
                cur = row.get(col)
                if pd.isna(cur):
                    cur = None
                else:
                    break

        # if still none, try filename
        if cur is None and "filename" in df.columns:
            cur = row["filename"]

        # normalize cur to full path if it's a filename only
        if cur is None:
            continue

        # convert to str and normalize slashes
        cur_s = str(cur).replace("\\", os.sep).replace("/", os.sep)

        # check existence
        if os.path.exists(cur_s):
            continue  # ok

        # not found
        missing_before += 1
        basename = os.path.basename(cur_s)
        found = find_file_for_basename(basename, search_root)
        if found:
            replaced.append((i, cur_s, found))
            if not dry_run:
                # update the best-known image_path column
                if "image_path" in df.columns:
                    df.at[i, "image_path"] = found
                elif "image_path_y" in df.columns:
                    df.at[i, "image_path_y"] = found
                elif "image_path_x" in df.columns:
                    df.at[i, "image_path_x"] = found
                else:
                    # fallback: create image_path column
                    df.at[i, "image_path"] = found

    if not dry_run and replaced:
        # backup original csv
        backup = str(path) + BACKUP_SUFFIX
        print(f"Backing up {path} -> {backup}")
        path.rename(backup)
        df.to_csv(path, index=False)
        print(f"Saved fixed CSV to {path}")

    return missing_before, replaced

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Don't save changes; only report")
    parser.add_argument("--apply", action="store_true", help="Apply changes to files")
    parser.add_argument("--search_root", default=SEARCH_ROOT, help="Root directory to search for images")
    args = parser.parse_args()

    if args.dry_run and args.apply:
        print("Use either --dry_run or --apply, not both.")
        sys.exit(1)
    do_apply = args.apply
    print("Search root:", args.search_root)
    total_missing = 0
    total_replacements = 0
    for csv in TARGET_CSVS:
        print("\nProcessing", csv)
        missing_before, repl = process_csv(csv, args.search_root, dry_run=(not do_apply))
        total_missing += missing_before
        total_replacements += len(repl)
        if repl:
            print(f"  {len(repl)} replacements (first 10 shown):")
            for i, (row_idx, old, new) in enumerate(repl[:10]):
                print(f"    row {row_idx}:")
                print(f"      old: {old}")
                print(f"      new: {new}")
        else:
            print("  No replacements found for this CSV")

    print("\nSummary:")
    print("  Total missing entries encountered:", total_missing)
    print("  Total replacements found:", total_replacements)
    if do_apply:
        print("Applied changes (backups created with suffix", BACKUP_SUFFIX + ")")
    else:
        print("Dry-run complete. Rerun with --apply to save fixes.")

if __name__ == "__main__":
    main()
