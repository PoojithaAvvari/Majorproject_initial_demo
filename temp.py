
import pandas as pd
p = "data/train_labeled.csv"
df = pd.read_csv(p)
# find rows where referenced image file does not exist
import os
bad_idx = []
for i,row in df.iterrows():
    # prefer image_path, fallback to variants
    cur = None
    for c in ("image_path","image_path_y","image_path_x"):
        if c in df.columns:
            cur = row.get(c)
            if pd.notna(cur):
                break
    if cur is None:
        cur = row.get("filename", "")
    cur_s = str(cur).replace("\\", os.sep).replace("/", os.sep)
    if not cur_s or not os.path.exists(cur_s):
        bad_idx.append(i)

print("Missing rows:", bad_idx)
if bad_idx:
    # backup
    import shutil
    shutil.copy(p, p + ".bak")
    df2 = df.drop(index=bad_idx).reset_index(drop=True)
    df2.to_csv(p, index=False)
    print("Saved cleaned", p, " (backup created at", p + ".bak )")
else:
    print("No missing files found; nothing changed.")
