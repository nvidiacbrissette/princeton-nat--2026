# NVIDIA NeMo Agent Toolkit — Training Assets

A hands-on training program for building, evaluating, and optimizing AI agents using the [NVIDIA NeMo Agent Toolkit (NAT)](https://github.com/NVIDIA/NeMo-Agent-Toolkit).

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **NVIDIA API Key** — free from [build.nvidia.com](https://build.nvidia.com/settings/api-keys)
- **No GPU required** — all inference uses public endpoints on build.nvidia.com

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/nat-training-assets.git
cd nat-training-assets
```

### API Key Setup

All notebooks load API keys from a `.env` file at the project root. You can either:

1. **Copy and fill out the example file:**

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

2. **Or start with `module-1-fundamentals/01_Introduction.ipynb`**, which includes an interactive API Key Setup section that creates the `.env` file for you.

## Training Modules

### Module 1 — Fundamentals (Beginner)

> `module-1-fundamentals/`

Get started with AI agents. Build your first agent from scratch and survey the major frameworks.

| Notebook | Topics |
|----------|--------|
| `01_Introduction.ipynb` | Course overview, module structure, prerequisites |
| `02_Build_Your_First_Agent.ipynb` | RAG agent with LangGraph, Tavily search, FAISS vector store, Gradio UI |
| `03_Surge_of_Agents.ipynb` | Comparing OpenAI, LangChain, LangGraph, and CrewAI |

### Module 2 — NeMo Agent Toolkit (Intermediate)

> `module-2-nemo-agent-toolkit/`

Learn the NeMo Agent Toolkit: build custom workflows, evaluate agents, set up observability, and run the optimizer.

| Notebook | Topics |
|----------|--------|
| `01_NeMo_Agent_Toolkit_First_Contact.ipynb` | NAT CLI, available components, configuration basics |
| `02_Build_A_Math_Pro_Agent.ipynb` | Custom tools, ReAct agent workflow, YAML config, `nat run` and `nat serve` |
| `03_Evaluation_Observability_And_Optimization.ipynb` | RAGAS evaluation, Phoenix observability, `nat optimize` intro |

#### Deep Dive (Advanced)

> `module-2-nemo-agent-toolkit/deep-dive/`

A production-grade extension of Module 2's eval/observability/optimization notebook, using a real-world email phishing analyzer workflow.

| Notebook | Topics |
|----------|--------|
| `01_Evaluation.ipynb` | Custom evaluators, RAGAS + trajectory metrics, dataset design |
| `02_Profiling.ipynb` | Token analysis, latency profiling, runtime forecasting, bottleneck identification |
| `03_Optimizing.ipynb` | Optuna numeric tuning, prompt genetic algorithms, Pareto trade-offs |

### Module 3 — Advanced Workflows (Advanced)

> `module-3-advanced-workflows/`

Build advanced agentic pipelines with RAG and multi-framework orchestration.

| Notebook | Topics |
|----------|--------|
| `01_RAG_It_Is_Ragtime.ipynb` | RAG with NAT, LlamaIndex, FAISS, NVIDIA reranking |
| `02_Surge_of_Agents_with_NeMo_Agent_Toolkit.ipynb` | Multi-framework integration (OpenAI, LangGraph, CrewAI) in a single NAT workflow |

## Technical Notes

- Each module has its own `pyproject.toml` — run `uv sync` to install dependencies and register the Jupyter kernel. After running `uv sync`, select the registered kernel in JupyterLab (refresh the page if already open) or VS Code (run **Developer: Reload Window** if the kernel doesn't appear)
- The deep-dive under Module 2 has a separate `pyproject.toml` with additional dependencies for the phishing analyzer
- All notebooks use `python-dotenv` to load API keys from the root `.env` file
- No GPU is required — all LLM inference goes through NVIDIA NIM endpoints on build.nvidia.com
