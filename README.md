# AegisRAG — Reproducibility Package

Supplementary code and data for *"AegisRAG: No-code configuration and
calibrated abstention for enterprise retrieval-augmented generation."*

This package reproduces the experiments reported in the paper: the
calibrated reranker-margin abstention head, the selective-QA evaluation,
and the AegisRAG-Enterprise benchmark.

## Contents

```
code/        Python source (pipeline, systems, calibration, evaluation, analysis)
data/        AegisRAG-Enterprise benchmark (authored by the research team)
  policy/    University-policy documents
  itdocs/    IT documentation
  hr/        HR documents
  questions/ benchmark.json — the enterprise QA pairs (answerable + unanswerable)
results/     Recorded experimental outputs (JSON) used to build the paper's tables/figures
requirements.txt
```

Public benchmarks are **not** redistributed here. Obtain them from their
canonical sources and place them under `data/`:

- **SQuAD v2**  — `data/squad/dev-v2.0.json` from https://rajpurkar.github.io/SQuAD-explorer/
- **NFCorpus, SciFact, FiQA** — test splits from the BEIR distribution
  (https://github.com/beir-cellar/beir)

## Environment

Python 3.11+ (developed on 3.13). Install dependencies:

```
pip install -r requirements.txt
```

The core retrieval/abstention pipeline and the open-weight generator
(FLAN-T5-Base) run **without any API keys**.

### Optional: frontier-model reruns

`code/run_llm_frontier.py` is an optional drop-in runner that swaps a
frontier model into the generation role for provider-grade reruns. It
reads credentials from environment variables (never hard-coded):

```
ANTHROPIC_API_KEY   # for --provider anthropic (default model: claude-sonnet-4-6)
OPENAI_API_KEY      # for --provider openai    (default model: gpt-4o)
```

The same models are used for the LLM-as-judge quality scores reported in
the paper. These reruns are optional; all headline selective-QA results
use the open-weight FLAN-T5-Base reference generator.

## Reproducing the main results

```
# 1. Build the AegisRAG-Enterprise benchmark from the source documents
python code/build_benchmark.py

# 2. Main selective-QA evaluation (retrieval + calibrated abstention)
python code/run_main.py

# 3. Component / ablation studies
python code/run_ablation.py

# 4. Aggregate and analyse
python code/analyze.py
```

Outputs are written to `results/`. The shipped `results/*.json` are the
exact records used for the tables and figures in the paper, so analysis
scripts can be run against them directly.

## Notes on faithful reporting

- Numbers in the paper are produced by the scripts above; `results/`
  contains the recorded runs.
- Generator and judge models are reported as used (FLAN-T5-Base for the
  reproducible reference; Llama 3, Claude Sonnet 4.6, and GPT-4o/GPT-4.5
  for the deployed/frontier validation and LLM-as-judge protocols).
