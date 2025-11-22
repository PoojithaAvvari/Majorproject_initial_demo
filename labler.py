# src/labeler.py
"""
Labeler for IU Chest X-ray reports.
Reads a metadata CSV (expects columns like uid, filename, image_path, findings, impression)
and produces a CSV with `labels_vec`, `uncertain_vec`, and `labels_readable`.

Run:
    python src/labeler.py --in data/metadata.csv --out data/metadata_labeled.csv

Outputs:
    - data/metadata_labeled.csv
    - data/label_names.json
"""

import re
import json
import argparse
from tqdm import tqdm
import pandas as pd
import os

# ------------------------
# Label set (modify as needed)
# ------------------------
LABELS = [
    "Cardiomegaly",
    "Effusion",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Edema",
    "Pneumothorax",
    "Mass",
    "Nodule",
    "Fracture",
    "No_Finding"
]

# Expand or tweak patterns for your dataset (radiology phrasing)
LABEL_PATTERNS = {
    "Cardiomegaly": [r"\bcardiomegaly\b", r"\benlarged heart\b", r"\bincreased cardiac silhouette\b"],
    "Effusion": [r"\beffusion\b", r"\bpleural effusion\b", r"\bsmall effusion\b", r"\blarge effusion\b"],
    "Consolidation": [r"\bconsolidation\b", r"\bairspace consolidation\b"],
    "Pneumonia": [r"\bpneumonia\b", r"\binfectious process\b", r"\bcommunity[- ]acquired pneumonia\b"],
    "Atelectasis": [r"\batelectasis\b", r"\bsubsegmental atelectasis\b"],
    "Edema": [r"\bedema\b", r"\binterstitial edema\b", r"\bcardiogenic pulmonary edema\b"],
    "Pneumothorax": [r"\bpneumothorax\b", r"\bvisceral pleural line\b", r"\bpartial pneumothorax\b"],
    "Mass": [r"\bmass\b", r"\bopacity suspicious for mass\b"],
    "Nodule": [r"\bnodule\b", r"\brounded opacity\b", r"\bsolitary pulmonary nodule\b"],
    "Fracture": [r"\bfracture\b", r"\bfx\b", r"\bbroken rib\b"],
    "No_Finding": [r"\bno acute cardiopulmonary process\b", r"\bno acute findings\b", r"\bno acute abnormality\b", r"\bno acute disease\b", r"\bno acute cardiopulmonary disease\b"]
}

# Negation and uncertainty keywords (simple window-based detection)
NEGATION_TERMS = [
    "no", "no evidence of", "without", "not", "absence of", "unlikely", "negative for", "free of"
]
UNCERTAINTY_TERMS = [
    "possible", "probable", "suggests", "likely", "cannot exclude", "may represent", "suspicious for", "questionable", "could be"
]

# Compile regexes
COMPILED_LABEL_REGEX = {
    lab: [re.compile(p, flags=re.I) for p in pats]
    for lab, pats in LABEL_PATTERNS.items()
}
NEGATION_REGEX = re.compile(r"\b(" + r"|".join([re.escape(x) for x in NEGATION_TERMS]) + r")\b", flags=re.I)
UNCERTAINTY_REGEX = re.compile(r"\b(" + r"|".join([re.escape(x) for x in UNCERTAINTY_TERMS]) + r")\b", flags=re.I)

def is_negated(text, match_start, window=60):
    start = max(0, match_start - window)
    ctx = text[start:match_start].lower()
    return bool(NEGATION_REGEX.search(ctx))

def is_uncertain(text, match_start, window=60):
    start = max(0, match_start - window)
    ctx = text[start:match_start].lower()
    return bool(UNCERTAINTY_REGEX.search(ctx))

def label_text(text, labels=LABELS):
    """
    Return (labels_vec, uncertain_vec)
    labels_vec: multi-hot list of ints (1 = positive, 0 = negative)
    uncertain_vec: list of ints (1 = uncertain mention, 0 = certain/none)
    """
    text = (text or "").replace("\n", " ").strip()
    labels_vec = [0] * len(labels)
    uncertain_vec = [0] * len(labels)

    # Short-circuit No_Finding if explicit
    for pat in COMPILED_LABEL_REGEX.get("No_Finding", []):
        if pat.search(text):
            idx = labels.index("No_Finding")
            labels_vec[idx] = 1
            return labels_vec, uncertain_vec

    # Search for each label
    for i, lab in enumerate(labels):
        if lab == "No_Finding":
            continue
        patterns = COMPILED_LABEL_REGEX.get(lab, [])
        found_any = False
        found_uncertain = False
        for pat in patterns:
            for m in pat.finditer(text):
                found_any = True
                s = m.start()
                if is_negated(text, s):
                    # skip negated mentions
                    continue
                if is_uncertain(text, s):
                    found_uncertain = True
                    # continue scanning to see if there is a CERTAIN mention elsewhere
                    continue
                # definite positive mention
                labels_vec[i] = 1
                uncertain_vec[i] = 0
                break
            if labels_vec[i] == 1:
                break
        # If no definite positive but uncertain mention(s) exist -> mark uncertain
        if labels_vec[i] == 0 and found_any and found_uncertain:
            uncertain_vec[i] = 1

    return labels_vec, uncertain_vec

def process_csv(in_csv, out_csv, text_cols_preference=None):
    """
    in_csv -> out_csv with added columns:
      - labels_vec (stringified list)
      - uncertain_vec (stringified list)
      - labels_readable (semicolon-separated)
    """
    df = pd.read_csv(in_csv)
    print(f"Loaded {in_csv} rows={len(df)} columns={list(df.columns)}")

    # prefer these text columns in order (user may not have all)
    text_cols_preference = text_cols_preference or ["impression", "findings", "image", "Problems"]

    labels_col = []
    uncertain_col = []
    readable_col = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # pick first available text field
        text = None
        for c in text_cols_preference:
            if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
                text = str(row[c])
                break
        labels_vec, uncertain_vec = label_text(text)
        labels_col.append(labels_vec)
        uncertain_col.append(uncertain_vec)
        readable = ";".join([LABELS[i] for i, v in enumerate(labels_vec) if v]) or "None"
        readable_col.append(readable)

    df["labels_vec"] = labels_col
    df["uncertain_vec"] = uncertain_col
    df["labels_readable"] = readable_col

    # Save as csv; ensure lists are stringified so CSV can store them
    df.to_csv(out_csv, index=False)
    print(f"Wrote labeled CSV to {out_csv}")

    # Save label order for downstream use
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    label_names_path = os.path.join(os.path.dirname(out_csv), "label_names.json")
    with open(label_names_path, "w") as f:
        json.dump(LABELS, f, indent=2)
    print("Saved label names to", label_names_path)
    return out_csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_csv", required=True, help="Input metadata CSV")
    parser.add_argument("--out", dest="out_csv", default="data/metadata_labeled.csv", help="Output labeled CSV")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    in_csv = args.in_csv
    out_csv = args.out_csv

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    process_csv(in_csv, out_csv)
