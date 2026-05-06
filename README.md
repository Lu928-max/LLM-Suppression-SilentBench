readme_full = """# LLM-Suppression-SilentBench

## Beyond the Chosen Token: Analyzing Output Suppression in Large Language Models

## Overview
SilentBench is a comprehensive benchmark for studying output suppression patterns in Large Language Models. This repository contains all code and scripts required to fully reproduce the results, figures, and tables in our paper.

## Requirements
Python 3.9+ and a CUDA-compatible GPU (minimum 16GB VRAM recommended).

```bash
pip install -r requirements.txt
```

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Login to HuggingFace
```bash
huggingface-cli login
```

Request access to:
- meta-llama/Llama-3.1-8B and meta-llama/Llama-3.1-8B-Instruct
- google/gemma-2b and google/gemma-2b-it

## Step 3: Run Experiments
```bash
cd experiments/01_OPT_base_vs_IML && python run_opt.py
cd experiments/02_Gemma_base_vs_Instruct && python run_gemma.py
cd experiments/03_Llama_base_vs_Instruct && python run_llama.py
cd experiments/04_Mistral_base_vs_Instruct && python run_mistral.py
```

## Step 4: Reproduce Table 1
```bash
cd analysis && python significance_tests.py
```
Output: results/final_significance_table.csv

## Step 5: Reproduce All Figures
```bash
cd analysis && python visualizations.py
```

Figures saved to results/figures/:
- Figure 1: base_vs_rlhf_20k.png
- Figure 2: pca_tsne_all4models.png
- Figure 3: cross_model_comparison.png
- Figure 4: grouped_bar_significance.png
- Figure 5: error_predictor_roc.png

## Step 6: Error Predictor
```bash
cd analysis && python error_predictor.py
```

## Dataset: SilentBench v1.0
- Total records: 35,000
- OPT: 20,000 records
- Gemma: 5,000 records
- Llama: 5,000 records
- Mistral: 5,000 records

## Key Results
| Model | Safety d | Factual d | Contested d | Knowledge d | Creative d |
|-------|----------|-----------|-------------|-------------|------------|
| OPT | 0.26*** | 0.007 ns | 0.149*** | 0.125*** | 0.054*** |
| Gemma | 1.718*** | 1.403*** | 1.310*** | 1.272*** | 0.320*** |
| Llama | 0.617*** | 1.114*** | 0.503*** | 0.600*** | 0.159*** |
| Mistral | 0.834*** | 1.491*** | 0.512*** | 0.537*** | 0.040 ns |

## Hardware Requirements
- GPU: NVIDIA A100 40GB recommended
- Minimum: NVIDIA T4 16GB
- RAM: 16GB minimum
- Storage: 50GB free

## Citation
To be added upon publication.

## License
MIT License."""

with open("/drive/MyDrive/NeurIPS/README.md", "w") as f:
    f.write(readme_full)
print("Saved README.md")
