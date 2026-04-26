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

I recommend you to run on Kaggle, which support GPU T4 x 2. This is the full version I used for video demonstration. Please access the following link for better inspection \[https://www.kaggle.com/code/thinhdo05/23120089](23120089-Kaggle-Demo)

Just run all the cell to see the results.

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
These are the hyperparameters I use to fine-tune the model, following the recommended instruction on Unsloth.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Base model | unsloth/llama-3-8b-bnb-4bit | Strong pretrained LLM, efficient with 4-bit quantization |
| LoRA rank (r) | 16 | Balance between trainable params and performance |
| LoRA alpha | 16 | Standard setting, equal to r |
| Batch size | 4 | Fits within Kaggle T4 16GB VRAM |
| Gradient accumulation | 4 | Effective batch size = 16 |
| Epochs | 2 | Sufficient for small dataset, avoids overfitting |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Optimizer | adamw_8bit | Memory efficient, recommended by Unsloth |
| Max sequence length | 2048 | Covers all banking intent messages |

## Results



| Metric | Value |
|--------|-------|
| Test Accuracy | 99.25% (397/400) |

### Why is accuracy so high?

The high accuracy is expected and valid for the following reasons:

1. **Reduced number of classes** — Only 10 out of 77 intents were used.
   Fewer classes means simpler decision boundaries for the model.

2. **Semantically distinct intents** — The 10 chosen intents have very
   different vocabulary patterns (e.g. `exchange_rate` vs `activate_my_card`),
   making them easy to distinguish.

3. **Strong base model** — LLaMA-3 8B is a powerful pretrained model that
   already understands language well. Fine-tuning it on even a small dataset
   for a simple task converges quickly.

4. **Sufficient training data** — ~125 samples per class is enough for
   fine-tuning a pretrained LLM on a classification task this simple.

5. **This is a subset result** — Accuracy would likely be lower on the
   full 77-class BANKING77 benchmark, which is a much harder problem.


## Video Demonstration

\[Link to demo video](VideoDemo)

