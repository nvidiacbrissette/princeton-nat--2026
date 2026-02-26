# Deep Dive: Evaluation, Profiling & Optimization

This deep-dive series extends the concepts introduced in [03_Evaluation_Observability_And_Optimization.ipynb](../03_Evaluation_Observability_And_Optimization.ipynb), taking them to production depth using a real-world **Email Phishing Analyzer** workflow.

## Prerequisites

- Complete Module 2's notebooks first, especially `03_Evaluation_Observability_And_Optimization.ipynb`
- NVIDIA API Key set in the root `.env` file

## Setup

```bash
cd module-2-nemo-agent-toolkit/deep-dive
uv sync
uv pip install -e .
uv run jupyter lab
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_Evaluation.ipynb` | Production-grade evaluation with RAGAS metrics, trajectory analysis, and custom evaluators |
| `02_Profiling.ipynb` | Token usage analysis, latency profiling, runtime forecasting, and bottleneck identification |
| `03_Optimizing.ipynb` | Multi-objective optimization with Optuna numeric tuning and prompt genetic algorithms |

## What Makes This Different from Module 2?

Module 2's notebook 03 introduces evaluation, observability, and optimization using a simple math agent with built-in evaluators. This deep dive:

- Uses a **production-grade email phishing analyzer** with 5 specialized tools
- Teaches you to **build custom evaluators** from scratch
- Covers **advanced profiling** (token uniqueness, concurrency analysis, cost forecasting)
- Demonstrates **prompt optimization** via genetic algorithms
- Explores **Pareto trade-offs** across multiple competing objectives
