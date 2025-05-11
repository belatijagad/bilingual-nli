# bilingual-nli

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Training
```bash
torchrun --nproc_per_node=<number_of_gpus> --master_port=<your_chosen_port> main.py train --config_file config.yml
```

Predict
```bash
python main.py predict --config_file config.yml
```

Predict with specific model path
```bash
python main.py predict --config_file config.yml --model_path_for_predict ./model_directory/
```