
## Reproducing Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

Configure API keys in `config/config2.yaml`:

### Run Experiments

Run experiments on HumanEval:

```bash
# HumanEval
python -m experiments.run_main --dataset HumanEval --sample 4 --round 1 --exec_model_name "gpt-4o-mini" --latency_weights "0.005" --lr 0.01 --normalize_rewards
```

### Results

Raw CSV results are saved to:

```
maas/ext/maas/scripts/optimized/latency_experiments/latency_weight_0_005/HumanEval/test/round_1/*.csv
```
