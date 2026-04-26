# Banking Intent Classification with Unsloth

Fine-tuning LLaMA-3 8B for banking intent detection using the BANKING77 dataset.

## Project Structure

```
banking-intent-unsloth/
|---scripts/
|      |-- train.py
|      |-- inference.py
|      |-- preprocess_data.py
|      |-- evaluate.py
|---configs/
|      |-- train.yaml
|      |-- inference.yaml
|---sample_data/
|      |-- train.csv
|      |-- test.csv
|--- train.sh
|--- inference.sh
|--- requirements.txt
|--- README.md
```


## Dataset

A 10-class subset of [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), containing the following intents:

- `activate_my_card`, `age_limit`, `card_arrival`, `change_pin`, `exchange_rate`
- `lost_or_stolen_card`, `passcode_forgotten`, `request_refund`, `terminate_account`, `transfer_timing`

| Split   | Samples |
|---------|---------|
| Train   | 1248    |
| Test    | 400     |

## Environment Setup

### Requirements

> **Important:** Training requires a CUDA-capable GPU.  
> Unsloth does not support CPU-only or Windows local machines.  
> It is recommended to run training on **Kaggle** (free T4 GPU) or **Google Colab**.

### Recommended Platforms

| Platform        | GPU          | Free |
|----------------|--------------|------|
| Kaggle         | T4 16GB      | Yes  |
| Google Colab   | T4 / V100    | Yes  |
| Local machine  | NVIDIA GPU required | —    |

### On Kaggle (recommended)

Run the first notebook cell to install dependencies:

```bash
pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers peft accelerate bitsandbytes
pip install transformers==4.51.3 trl==0.18.2 datasets==3.4.1
```


## How to Run



### Step 1 — Preprocess data

```bash

python scripts/preprocess_data.py

```

Downloads BANKING77, filters 10 intents, cleans text, and saves to `sample_data/`.



### Step 2 — Train

```bash

bash train.sh

```
or directly:

```bash

python scripts/train.py

```

Fine-tunes LLaMA-3 8B with LoRA (r=16) for 2 epochs. Saves checkpoint to `banking\_model\_checkpoint/`.



### Step 3 — Inference

```bash

bash inference.sh

```

or directly:

```bashh

python scripts/inference.py

python scripts/evaluate.py

```



## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | unsloth/llama-3-8b-bnb-4bit |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Optimizer | adamw\_8bit |
| Max sequence length | 2048 |



## Results



| Metric | Value |
|--------|-------|
| Test Accuracy | 99.25% (397/400) |



## Video Demonstration

\[Link to demo video](VideoDemo)

