
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch
import torch.nn as nn
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
import pandas as pd
from datasets import Dataset
import json
import yaml

# Load configs
with open("configs/train.yaml") as f:
    cfg = yaml.safe_load(f)

# Patch for Unsloth + transformers 5.5.0 + trl 0.24.0
_orig_compute_loss = Trainer.compute_loss

def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.pop("labels") if "labels" in inputs else None
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    if loss is None:
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
    return (loss, outputs) if return_outputs else loss

Trainer.compute_loss = _patched_compute_loss

# Load model
max_seq_length = cfg["max_seq_length"]

model, tokenizer = FastLanguageModel.from_pretrained(
    cfg["model_name"],
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=cfg["load_in_4bit"],
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora_r"],
    target_modules=cfg["lora_target_modules"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=cfg["seed"],
)

# Load data
with open(cfg["label_mapping_path"]) as f:
    mapping = json.load(f)
    id_to_name = {int(k): str(v) for k, v in mapping["id_to_name"].items()}

train_df = pd.read_csv(cfg["train_data_path"])
train_df = train_df.dropna(subset=["text", "label"])
train_df["label"] = train_df["label"].astype(int)
train_df = train_df[train_df["label"].isin(id_to_name)].reset_index(drop=True)
print(f"Training samples: {len(train_df)}")

dataset = Dataset.from_pandas(train_df, preserve_index=False)

# Tokenizer
def tokenize(ex):
    prompt = (
        "Question: What is the banking intent of the following message?\n"
        f"Message: {ex['text']}\n"
        "Answer: "
    )
    answer = id_to_name[int(ex["label"])] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=True,  truncation=True, max_length=max_seq_length).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_seq_length - len(prompt_ids)).input_ids

    input_ids = (prompt_ids + answer_ids)[:max_seq_length]
    labels    = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_length]
    return {"input_ids": input_ids, "labels": labels}

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    dataset_text_field=None,
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    ),
    args=TrainingArguments(
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=cfg["logging_steps"],
        optim=cfg["optimizer"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=cfg["seed"],
        output_dir=cfg["output_dir"],
        report_to="none",
        save_steps=cfg["save_steps"],
        remove_unused_columns=False,
    ),
)

trainer.train()

model.save_pretrained(cfg["checkpoint_dir"])
tokenizer.save_pretrained(cfg["checkpoint_dir"])
print(f"Done. Checkpoint saved to '{cfg['checkpoint_dir']}'")
