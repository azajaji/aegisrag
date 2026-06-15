# AutoRAG Revision v2 — Reproducibility Manifest

This directory contains the additional artefacts produced by the
revision pass that extended the paper with selective-QA decomposition,
an expanded baseline (`RAG+Rerank+Abstain`), threshold-sweep diagnostics,
calibration AUROC/AUPRC analysis, repeated-split robustness, cost
sensitivity, latency distributions, paired statistical tests with
Benjamini–Hochberg FDR correction, and an Enterprise dataset audit.

## Environment

- Python: 3.13.9 (system)
- NumPy / SciPy / Matplotlib via Anaconda 2025.11
- LaTeX: MiKTeX-pdfTeX 4.23 (MiKTeX 25.12)
- Hardware: single CPU host; deployed-system validation in §5.9 of the
  paper was executed by the IS498 senior project team on a workstation
  with AMD Ryzen 5 5600X, NVIDIA RTX 3060 Ti, 16 GB RAM (source: student
  report §7.2)
- Random seed: 20260511 (master); repeated splits use seeds 20260511,
  20260512, 20260513, 20260514, 20260515

## Reproduce

From `paper/code/`:

```bash
# (1) Recompute headline metrics from existing per-question outputs
python revision_analyze.py

# (2) Regenerate revision_v2 figures
python revision_figures.py

# (3) Compile the manuscript
cd ../tex
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

`revision_analyze.py` is fully deterministic — it derives all
selective-QA metrics from the existing `results/main_squad.json`,
`results/main_enterprise.json`, `results/main_nfcorpus.json`,
`results/llm_squad.json`, and `results/llm_enterprise.json` files,
without re-running retrieval, reranking, or LLM inference. The
RAG+Rerank+Abstain baseline is derived by applying the calibrated
abstention rule to RAG+Rerank's per-question cross-encoder logits and
margins, with thresholds selected on the 40 % calibration split.

## Outputs

### Section 1: Reproduction

- `reproduction/` — placeholder for full re-runs of `run_main.py`,
  `run_llm.py`, `run_ablation.py`. The analyses in this revision pass
  use the existing committed per-question outputs unchanged; running
  the original scripts will reproduce them.

### Section 3: False-refusal metrics

- `abstention/false_refusal_metrics.csv` — coverage, false-refusal,
  refusal accuracy, hallucination, citation precision when answered,
  F1 (all answerable), F1 (answered only) for each system on each
  dataset's eval split.

### Section 4: Threshold sweep + calibration diagnostics

- `abstention/threshold_sweep_squad.csv` — 41 × 21 = 861 (τs, τm) pairs
  with answerable retention, unanswerable refusal, balanced abstention
  accuracy for AutoRAG on SQuAD-v2.
- `abstention/threshold_sweep_enterprise.csv` — same for Enterprise.
- `abstention/calibration_diagnostics.csv` — AUROC / AUPRC for top-1
  reranker score, margin, and the combined min-rule on each dataset.

Key finding: top-1 reranker score is a near-perfect unanswerable
detector on Enterprise (AUROC 0.995, AUPRC 0.921) and a moderate
detector on SQuAD-v2 (AUROC 0.683).

### Section 5: Expanded baselines

- `baselines/expanded_baseline_metrics.csv` — BM25, NaiveRAG,
  RAG+Rerank, **RAG+Rerank+Abstain (NEW)**, AutoRAG selective-QA
  metrics on both datasets.

Key finding: on the Enterprise eval split, RAG+Rerank+Abstain matches
AutoRAG on every selective-QA metric (refusal 0.947, false-refusal
0.030, citation-when-answered 0.981, F1-answered 0.331). On SQuAD-v2
AutoRAG retains a small edge in refusal (+0.034), citation-when-
answered (+0.037), and F1-answered (+0.022). This supports the
honest framing in §6.4 of the paper: the calibrated abstention head
is the dominant contributor to selective-QA gains; AutoRAG's broader
contribution is the no-code orchestration of that mechanism.

### Section 6: Repeated-split robustness

- `robustness/repeated_split_metrics.csv` — 5 seeds × (SQuAD-v2,
  Enterprise) × (BM25, NaiveRAG, RAG+Rerank, RAG+Rerank+Abstain,
  AutoRAG) × all selective-QA metrics.

Median refusal accuracy on Enterprise: 0.95 (IQR small); on SQuAD-v2:
0.66 (IQR ≈ 0.04).

### Section 7: Enterprise dataset audit

- `enterprise_dataset_audit/audit_report.md` — 305 questions, 275
  answerable, 30 unanswerable, 0 duplicates, all answerable have
  non-empty gold answers, all unanswerable have empty gold answers.
  Domain split: HR 100, IT 90, Policy 115.
- `enterprise_dataset_audit/question_distribution.csv` — domain counts.

### Section 9: Cost sensitivity

- `cost/cost_sensitivity.csv` — USD per 1{,}000 queries for each
  (dataset, system, model in {gpt-4o-mini, claude-haiku-4.5, gpt-4o,
  claude-sonnet-4.6}, context in {top-1, top-3, top-5}).

### Section 10: Latency

- `latency/component_latency_summary.csv` — mean / median / p90 / p95 /
  p99 / std / min / max per system per dataset.

Component-level latency instrumentation (separating embedding,
retrieval, rerank, abstention, generation) was not added in this pass;
the existing per-question `latency` field captures total end-to-end
latency on the CPU host.

### Section 13: Statistical tests

- `statistics/statistical_tests_full.csv` — Wilcoxon (for continuous
  metrics) and McNemar (for the binary `refusal_correct` indicator)
  paired tests, AutoRAG vs each of BM25 / NaiveRAG / RAG+Rerank, with
  Cohen's d_z and Benjamini–Hochberg FDR-adjusted p-values within each
  (dataset, metric) family.

### Section 14: Error analysis

- `error_analysis/false_refusals.csv` — answerable queries that
  AutoRAG refused (up to 40 samples per dataset).
- `error_analysis/false_answers_unanswerable.csv` — unanswerable
  queries that AutoRAG answered.
- `error_analysis/citation_failures.csv` — answered answerable queries
  with citation accuracy 0.

## Deferred / Out-of-scope

The following items from the revision-instructions document were
explicitly out of scope for this single-session pass and remain
future work:

- **§2 LLM-head decomposed reruns (3 modes × systems × datasets)** —
  Requires re-running FLAN-T5-Base in three configurations
  (no abstention, prompt-only refusal, calibrated abstention). Each
  mode is a multi-hour CPU job; deferred.
- **§5 HybridRAG and HybridRAG+Rerank** as actually-run baselines —
  Requires re-indexing + retrieval with new flag combinations.
  RAG+Rerank+Abstain is included as a derived baseline since its
  per-question reranker scores are already captured.
- **§8 Provider-grade LLM evaluation (GPT-4o, Claude, Gemini)** — No
  API keys configured in this environment.
- **§12 In-person controlled usability study** — The 19-participant
  survey from the IS498 student team (paper §5.10) is the available
  usability evidence; running a fresh controlled study is out of
  scope.
- **Component-level latency instrumentation** — Would require code
  changes to `systems.py` to time embedding / retrieval / rerank /
  abstention / generation separately.

## What changed in the manuscript

New / revised paper content (paper/tex/main.tex):

- §5.3 **Selective-QA Decomposition and an Expanded Baseline** (new) —
  Introduces `RAG+Rerank+Abstain`, the new selective-QA decomposition
  table, threshold-sweep Pareto curve, calibration AUROC/AUPRC
  diagnostics, repeated-split robustness boxplots.
- §6.4 **Where the Selective-QA Gain Comes From** (new) — Honest
  Scenario-C interpretation: on Enterprise the calibrated abstention
  head is the dominant contributor; AutoRAG's broader value is no-code
  orchestration of the mechanism rather than any single algorithmic
  novelty.
- §5.12 **Cost and Latency** (expanded) — adds cost-sensitivity grid
  (four LLM rates × three context sizes) and latency distribution
  boxplots on the log axis.
- §5.4 **Rubric-based answer audit** (reframed) — single-rater
  diagnostic, no longer claims to "corroborate" the headline results.

## What did not change

The headline retrieval, refusal, citation, and cost numbers are
unchanged — they were re-derived from the same per-question outputs
and match the reported values in summary.json. The Enterprise full-
benchmark size remains n=305 with 275/30 answerable/unanswerable
split, audited and confirmed in this pass.
