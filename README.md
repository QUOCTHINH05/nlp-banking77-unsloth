\# Banking Intent Classification with Unsloth



Fine-tuning LLaMA-3 8B for banking intent detection using the BANKING77 dataset.



\## Project Structure

banking-intent-unsloth/

|-- scripts/

|   |-- train.py

|   |-- inference.py

|   |-- preprocess\_data.py

|   |-- evaluate.py

|-- configs/

|   |-- train.yaml

|   |-- inference.yaml

|-- sample\_data/

|   |-- train.csv

|   |-- test.csv

|-- train.sh

|-- inference.sh

|-- requirements.txt

|-- README.md



\## Dataset

A 10-class subset of \[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking\_data),

containing the following intents:

\- activate\_my\_card, age\_limit, card\_arrival, change\_pin, exchange\_rate

\- lost\_or\_stolen\_card, passcode\_forgotten, request\_refund, terminate\_account, transfer\_timing



| Split | Samples |

|-------|---------|

| Train | 1248    |

| Test  | 400     |



\## Environment Setup



\### On Kaggle (recommended)

Run the first notebook cell to install dependencies:

```bash

pip install "unsloth\[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps xformers peft accelerate bitsandbytes

pip install transformers==4.51.3 trl==0.18.2 datasets==3.4.1

```



\### Local machine

```bash

pip install -r requirements.txt

```



\## How to Run



\### Step 1 — Preprocess data

```bash

python scripts/preprocess\_data.py

```

Downloads BANKING77, filters 10 intents, cleans text, and saves to `sample\_data/`.



\### Step 2 — Train

```bash

bash train.sh

\# or directly:

python scripts/train.py

```

Fine-tunes LLaMA-3 8B with LoRA (r=16) for 2 epochs. Saves checkpoint to `banking\_model\_checkpoint/`.



\### Step 3 — Inference

```bash

bash inference.sh

\# or directly:

python scripts/inference.py

python scripts/evaluate.py

```



\## Hyperparameters



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



\## Results



| Metric | Value |

|--------|-------|

| Test Accuracy | 99.25% (397/400) |



\## Video Demonstration

\[Link to demo video](YOUR\_GOOGLE\_DRIVE\_LINK\_HERE)

