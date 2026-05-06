# LLM-Suppression-SilentBench

## What Models Don't Say: A Systematic Study of Output Suppression in Large Language Models

### Overview
SilentBench is a benchmark for studying output suppression patterns in Large Language Models.
We study how RLHF/instruction tuning changes what models *almost* say but don't.

### Key Findings
- Suppression is perfectly consistent (std=0.000) — not random noise
- RLHF creates category-specific suppression signatures across all model families
- Effect strongest on safety (d=1.73) and factual_contested (d=1.49) categories
- Zero hard refusals in 1B-8B models — suppression is the primary alignment mechanism

### Dataset: SilentBench v1.0
- 35,000 records across 4 model families
- 5 prompt categories: safety, factual, factual_contested, knowledge_boundary, creative
- Models: OPT, Gemma, Llama 3.1, Mistral

### Model Pairs
| Experiment | Base Model | Instruction-Tuned |
|------------|-----------|-------------------|
| OPT | facebook/opt-1.3b | facebook/opt-iml-1.3b |
| Gemma | google/gemma-2b | google/gemma-2b-it |
| Llama | meta-llama/Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct |
| Mistral | mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-Instruct-v0.2 |

### Setup
```bash
pip install -r requirements.txt
```

### Running Experiments
```bash
cd experiments/01_OPT_base_vs_IML
python run_opt.py

cd experiments/02_Gemma_base_vs_Instruct
python run_gemma.py

cd experiments/03_Llama_base_vs_Instruct
python run_llama.py

cd experiments/04_Mistral_base_vs_Instruct
python run_mistral.py
```

### Results
See `results/complete_stats_table.csv` for full statistical results.

### Citation
[To be added upon publication]

### License
MIT
