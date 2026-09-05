import csv
import random
import re
from collections import defaultdict

SRC = "azerbaijani_toxicity_dataset_fixed.csv"
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # remainder (0.1) goes to test

random.seed(SEED)


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


with open(SRC, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

tox_idx = header.index("toxicity")

# group near-duplicate comments so they stay together in one split
groups = defaultdict(list)
for row in rows:
    key = normalize(row[0]) or row[0]
    groups[key].append(row)

# stratify groups by their majority toxicity label
by_label = defaultdict(list)
for key, group_rows in groups.items():
    label = round(sum(float(r[tox_idx]) for r in group_rows) / len(group_rows))
    by_label[label].append(group_rows)

train, val, test = [], [], []
for label, group_list in by_label.items():
    random.shuffle(group_list)
    n = len(group_list)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    for g in group_list[:n_train]:
        train.extend(g)
    for g in group_list[n_train:n_train + n_val]:
        val.extend(g)
    for g in group_list[n_train + n_val:]:
        test.extend(g)

for name, split in [("train", train), ("val", val), ("test", test)]:
    random.shuffle(split)
    with open(f"azerbaijani_toxicity_{name}.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(split)
    pos = sum(1 for r in split if r[tox_idx] == "1.0")
    print(f"{name}: {len(split)} rows, toxicity positive rate = {pos / len(split) * 100:.2f}%")
