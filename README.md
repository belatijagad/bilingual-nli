# bilingual-nli

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Flashattention (panjang umur orang baik)
```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.5/flash_attn-2.7.4.post1+cu124torch2.6-cp310-cp310-linux_x86_64.whl
```

## Usage
Training
```bash
python3 train.py
```

Predict
```bash
python infer.py --config_file config.yml
```

Predict with specific model path
```bash
python main.py predict --config_file config.yml --model_path_for_predict ./model_directory/
```

Submit
```bash
kaggle competitions submit -c natural-language-inference-dl-genap-2024-2025 -f submission.csv -m "Message"
```