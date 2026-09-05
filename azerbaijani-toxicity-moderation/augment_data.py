import csv
import random

SRC = "azerbaijani_toxicity_train.csv"
OUT = "azerbaijani_toxicity_train_augmented.csv"
SEED = 42

# Only boost genuinely rare labels. toxicity/insult are already well
# represented and don't need synthetic help.
RARE_LABELS = ["identity_attack", "obscene", "severe_toxicity", "sexual_explicit", "threat"]
TARGET_COUNT = 5000  # aim to bring each rare label's positive count up to this

random.seed(SEED)


def char_noise(word):
    if len(word) < 4:
        return word
    op = random.choice(["swap", "delete", "duplicate"])
    i = random.randrange(1, len(word) - 1)
    if op == "swap":
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if op == "delete":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def augment(text):
    words = text.split(" ")
    if len(words) > 3 and random.random() < 0.3:
        # random deletion of one non-edge word
        i = random.randrange(1, len(words) - 1)
        del words[i]
    if len(words) > 1 and random.random() < 0.3:
        # random adjacent word swap
        i = random.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    words = [char_noise(w) if random.random() < 0.3 else w for w in words]
    return " ".join(words)


with open(SRC, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

label_idx = {name: header.index(name) for name in RARE_LABELS}
new_rows = []

for label, col in label_idx.items():
    positives = [r for r in rows if r[col] == "1.0"]
    current = len(positives)
    deficit = max(0, TARGET_COUNT - current)
    if deficit == 0:
        print(f"{label}: already at {current}, skipping")
        continue
    sampled = [random.choice(positives) for _ in range(deficit)]
    for r in sampled:
        aug_row = list(r)
        aug_row[0] = augment(r[0])
        new_rows.append(aug_row)
    print(f"{label}: {current} -> target {TARGET_COUNT}, added {deficit} augmented rows")

with open(OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
    writer.writerows(new_rows)

print(f"\nOriginal train rows: {len(rows)}")
print(f"Augmented rows added: {len(new_rows)}")
print(f"Total: {len(rows) + len(new_rows)}")
print(f"Written to: {OUT}")
