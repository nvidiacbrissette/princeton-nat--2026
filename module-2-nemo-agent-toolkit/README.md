# Module 2: NeMo Agent Toolkit

Learn to build, evaluate, observe, and optimize agentic workflows using the NVIDIA NeMo Agent Toolkit (NAT).

## Prerequisites

- Complete Module 1 (Fundamentals)
- NVIDIA API Key set in the root `.env` file

## Setup

```bash
cd module-2-nemo-agent-toolkit
uv sync
uv run jupyter lab
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_NeMo_Agent_Toolkit_First_Contact.ipynb` | Introduction to the NAT CLI and available components |
| `02_Build_A_Math_Pro_Agent.ipynb` | Build a custom NAT workflow with math tools |
| `03_Evaluation_Observability_And_Optimization.ipynb` | Evaluate agents with RAGAS, set up Phoenix observability, and run the optimizer |

## Deep Dive

The `deep-dive/` directory contains an advanced, production-grade extension that applies evaluation, profiling, and optimization to a real-world email phishing analyzer. See [deep-dive/README.md](deep-dive/README.md) for details.
