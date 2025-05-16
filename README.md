# bilingual-nli

## Setup

### Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install FlashAttention

```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.5/flash_attn-2.7.4.post1+cu124torch2.6-cp310-cp310-linux_x86_64.whl
```

### Setup Environment Variables

Create `.env` file with this as the content

```bash
WANDB_TOKEN=
HF_TOKEN=
```

### Setup Kaggle

```bash
export KAGGLE_USERNAME=
export KAGGLE_TOKEN=
```

### Download Dataset

```bash
sudo apt-get install unzip
kaggle datasets download alifbintang/deeplearning-translated-processed-nli-data
unzip deeplearning-translated-processed-nli-data
mkdir data
mv processed_translated_train.csv processed_translated_test.csv data/
rm deeplearning-translated-processed-nli-data
```

## Usage

### Train

```bash
python3 train.py
```

### Inference

```bash
python3 inference.py \
  --model_path_or_id "your-username/your-trained-nli-model-id" \
  --input_csv_path "path/to/your/input_test_data.csv" \
  --output_verbose_csv "predictions_verbose.csv" \
  --output_submission_csv "submission.csv" \
  --config_path "config.yaml"
```

### Submit

```bash
kaggle competitions submit -c natural-language-inference-dl-genap-2024-2025 -f submission.csv -m "Commit message"
```