from unsloth import FastLanguageModel
import torch
import json
import yaml
import re


class IntentClassification:
    def __init__(self, model_path: str):
        """
        model_path: path to inference.yaml config file
        """
        # Load config
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        checkpoint        = config["model_checkpoint"]
        mapping_path      = config["label_mapping"]
        self.max_new_tokens = config.get("max_new_tokens", 16)
        max_seq_length    = config.get("max_seq_length", 2048)

        # Load label mapping
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        self.id_to_name = {int(k): str(v) for k, v in mapping["id_to_name"].items()}
        # Reverse mapping for validation
        self.valid_labels = set(self.id_to_name.values())

        # Load model & tokenizer from checkpoint
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            checkpoint,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)  # enable optimized inference
        self.model.eval()

    def __call__(self, message: str) -> str:
        """
        message: raw input text from user
        returns: predicted intent label (string)
        """
        prompt = (
            "Question: What is the banking intent of the following message?\n"
            f"Message: {message}\n"
            "Answer: "
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # greedy — deterministic for classification
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the prompt)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        predicted_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Clean up and validate against known labels
        predicted_label = predicted_text.lower().replace(" ", "_")
        if predicted_label not in self.valid_labels:
            # Fuzzy fallback: return closest match by overlap
            predicted_label = max(
                self.valid_labels,
                key=lambda l: len(set(predicted_label.split("_")) & set(l.split("_")))
            )

        return predicted_label


# Usage
if __name__ == "__main__":
    classifier = IntentClassification("configs/inference.yaml")

    test_messages = [
        "I need to activate my new card",
        "What is the exchange rate for USD to EUR?",
        "I forgot my passcode, how do I reset it?",
        "My card was stolen yesterday",
        "How long does a transfer take?",
    ]

    print("\n── Inference Results ──────────────────────────────")
    for msg in test_messages:
        label = classifier(msg)
        print(f"  Input : {msg}")
        print(f"  Intent: {label}")
        print()
