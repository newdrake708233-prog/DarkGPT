import os
import re

OUTPUT_FILE = "training.txt"
script_dir  = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, OUTPUT_FILE)

def clean(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

try:
    from datasets import load_dataset
except ImportError:
    print("Missing dependency.")
    exit(1)

dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
            split="train_sft[:50000]"
)

print(f"Downloaded — {len(dataset):,} conversations\n")
print(f"Converting and appending to {OUTPUT_FILE}...")

size_before = os.path.getsize(output_path) if os.path.exists(output_path) else 0
written     = 0
skipped     = 0

with open(output_path, "a", encoding="utf-8") as f:
    f.write("\n\nCONVERSATION TOPIC: UltraChat Natural Conversations\n\n")

    for row in dataset:
        messages = row.get("messages", [])

        if len(messages) < 2:
            skipped += 1
            continue

        for i in range(0, len(messages) - 1, 2):
            user = clean(messages[i].get("content", "").strip())
            ai = clean(messages[i + 1].get("content", "").strip())

            #Skip very long responses to keep training data balanced
            if user and ai and len(ai) < 1000:
                f.write(f"User: {user}\n")
                f.write(f"AI: {ai}\n\n")
                written += 1

size_after = os.path.getsize(output_path)
added_mb   = (size_after - size_before) / (1024 * 1024)
total_mb   = size_after / (1024 * 1024)

print(f"\nDone")
print(f"   Conversations written : {written:,}")
print(f"   Skipped               : {skipped:,}")
print(f"   Added to file         : {added_mb:.1f} MB")
print(f"   Total training.txt    : {total_mb:.1f} MB")
print(f"\nNow retrain with:  python train.py --data {OUTPUT_FILE} --steps 10000")