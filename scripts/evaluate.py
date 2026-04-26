from inference import IntentClassification
import pandas as pd
import json
from tqdm import tqdm

# Load test data
test_df = pd.read_csv("sample_data/test.csv")
test_df = test_df.dropna(subset=["text", "label"])
test_df["label"] = test_df["label"].astype(int)

with open("sample_data/label_mapping.json") as f:
    mapping = json.load(f)
id_to_name = {int(k): str(v) for k, v in mapping["id_to_name"].items()}

# Run inference
classifier = IntentClassification("configs/inference.yaml")

correct = 0
total = len(test_df)

for _, row in tqdm(test_df.iterrows(), total=total, desc="Evaluating"):
    true_label  = id_to_name[int(row["label"])]
    pred_label  = classifier(str(row["text"]))
    if pred_label == true_label:
        correct += 1

accuracy = correct / total * 100
print(f"\n── Test Set Accuracy ──────────────────────────────")
print(f"  Correct : {correct}/{total}")
print(f"  Accuracy: {accuracy:.2f}%")
