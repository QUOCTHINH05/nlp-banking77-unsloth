# Banking Intent Classification with Unsloth

Fine-tuning **LLaMA-3 8B** for banking intent detection using the BANKING77 dataset.

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

### Running on Kaggle (recommended)

1. The first cell is to install dependencies:

```bash
!pip install unsloth
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```

2. Clone the repository

```bash
!git clone https://github.com/QUOCTHINH05/nlp-banking77-unsloth.git
%cd nlp-banking77-unsloth
!ls -la
```

3. Run the preprocessing file

```bash
!python scripts/preprocess_data.py
````

4. Run the training file

```bash
!python scripts/train.py
```

5. Run the inference

```bash
!python scripts/inference.py
```

6. Run the evaluation

```bash
!python scripts/evaluate.py
```

7. (Optional) Testing Freely

In the `print` statement, replace the queries "I lost my card and need a new one" with your own question.

```bash
from scripts.inference import IntentClassification
clf = IntentClassification("configs/inference.yaml")
print(clf("I lost my card and need a new one"))
```

## The demo on Kaggle

I recommend you to run on Kaggle, which support GPU T4 x 2. This is the full version I used for video demonstration. Please access the following link for better inspection: 
[23120089-DemoKaggle](https://www.kaggle.com/code/thinhdo05/23120089)

Just run all the cell to see the results.

## The Video Demonstration
Here is the video's Google Drive link for demo: 
[VideoDemo](https://drive.google.com/file/d/1DsUICCQn3Pcqck5Ej2ep6t5TvMAhz3gd/view?usp=sharing)


## Hyperparameters
These are the hyperparameters I use to fine-tune the model, following the recommended instruction on [Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide).

| Parameter | Value | Reason |
|-----------|-------|--------|
| Base model | unsloth/llama-3-8b-bnb-4bit | Strong pretrained LLM, efficient with 4-bit quantization |
| LoRA rank (r) | 16 | Balance between trainable params and performance |
| LoRA alpha | 16 | Standard setting, equal to r |
| LoRA dropout | 0 | 0 is optimized, according to Unsloth guide |
| Batch size | 4 | Fits within Kaggle T4 16GB VRAM |
| Gradient accumulation | 4 | Effective batch size = 16 |
| Epochs | 2 | Sufficient for small dataset, avoids overfitting |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Optimizer | adamw_8bit | Memory efficient, recommended by Unsloth |
| Max sequence length | 2048 | Covers all banking intent messages |
| Regularization (weight decay) | 0.01 | Small number, don't use to large, 0.01 is recommended by Unsloth |
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

