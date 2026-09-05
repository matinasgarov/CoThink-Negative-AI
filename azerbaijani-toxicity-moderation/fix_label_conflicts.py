import csv
import re
from collections import defaultdict, Counter

SRC = "azerbaijani_toxicity_dataset.csv"
OUT = "azerbaijani_toxicity_dataset_fixed.csv"


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


with open(SRC, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

label_cols = header[1:]
groups = defaultdict(list)
for idx, row in enumerate(rows):
    key = normalize(row[0])
    if key:
        groups[key].append(idx)

fixed_groups = 0
fixed_rows = 0
ties_broken_positive = 0

for key, idxs in groups.items():
    if len(idxs) < 2:
        continue
    label_sets = set(tuple(rows[i][1:]) for i in idxs)
    if len(label_sets) <= 1:
        continue  # already consistent

    fixed_groups += 1
    new_labels = []
    for col_i in range(len(label_cols)):
        votes = Counter(rows[i][col_i + 1] for i in idxs)
        top_val, top_count = votes.most_common(1)[0]
        # tie: default to the positive ("1.0") label -- safer for a
        # toxicity-moderation dataset than silently under-labeling
        if len(votes) > 1 and list(votes.values()).count(top_count) > 1:
            top_val = "1.0" if "1.0" in votes else top_val
            ties_broken_positive += 1
        new_labels.append(top_val)

    for i in idxs:
        if rows[i][1:] != new_labels:
            fixed_rows += 1
        rows[i][1:] = new_labels

with open(OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Conflicting groups resolved: {fixed_groups}")
print(f"Rows whose labels changed: {fixed_rows}")
print(f"Tie-break votes (defaulted to positive): {ties_broken_positive}")
print(f"Written to: {OUT}")
