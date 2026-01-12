
## Reproducing Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

Configure API keys in `config/config2.yaml`.

### Dataset Setup

Please download the GSM8K, HumanEval, and MATH datasets and place them in the `lamas/ext/lamas/data` folder. The file structure should be organized as follows:

```
lamas/ext/lamas/data
├── gsm8k_train.jsonl
├── gsm8k_test.jsonl
└── ...
```

### Run Experiments

Run experiments on HumanEval:

```bash
# HumanEval
python -m experiments.run_main --dataset HumanEval --sample 4 --round 1 --exec_model_name "gpt-4o-mini" --latency_weights "0.005" --lr 0.01 --normalize_rewards
```

### Results

Raw CSV results are saved to:

```
lamas/ext/lamas/scripts/optimized/latency_experiments/latency_weight_0_005/HumanEval/test/round_1/*.csv
```

## Acknowledgements
Special thanks to [MaAS](https://github.com/bingreeky/MaAS) for providing the invaluable code and prompts that served as the foundation for this project.