import pandas as pd
import re
import os
import json

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data
TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
train_df = pd.read_csv(TRAIN_URL)
test_df = pd.read_csv(TEST_URL)

# Rename category to label
train_df = train_df.rename(columns={'category': 'label'})
test_df = test_df.rename(columns={'category': 'label'})

# 10 label selected for sampling
target_intent_names = [
    "activate_my_card",
    "age_limit",
    "card_arrival",
    "change_pin",
    "exchange_rate",
    "lost_or_stolen_card",
    "passcode_forgotten",
    "request_refund",
    "terminate_account",
    "transfer_timing"
]

# Filter by label
train_subset = train_df[train_df['label'].isin(target_intent_names)].copy()
test_subset = test_df[test_df['label'].isin(target_intent_names)].copy()

# Clean text
train_subset['text'] = train_subset['text'].apply(clean_text)
test_subset['text'] = test_subset['text'].apply(clean_text)

# Mapping intent name to 0-10
unique_intents = sorted(train_subset['label'].unique())
intent_to_id = {intent: i for i, intent in enumerate(unique_intents)}
id_to_intent = {i: intent for intent, i in intent_to_id.items()}

# Convert label to id
train_subset['label'] = train_subset['label'].map(intent_to_id)
test_subset['label'] = test_subset['label'].map(intent_to_id)

# Save file
os.makedirs('sample_data', exist_ok=True)
train_subset.to_csv('sample_data/train.csv', index=False)
test_subset.to_csv('sample_data/test.csv', index=False)

# Save mapping
with open('sample_data/label_mapping.json', 'w') as f:
    json.dump({
        "id_to_name": id_to_intent,
        "name_to_id": intent_to_id
    }, f, indent=2)

print(f"--- Kết quả Preprocessing ---")
print(f"Số lượng mẫu train: {len(train_subset)}")
print(f"Số lượng mẫu test: {len(test_subset)}")
print(f"Mapping nhãn (id -> tên): {id_to_intent}")
print(train_subset.head())
