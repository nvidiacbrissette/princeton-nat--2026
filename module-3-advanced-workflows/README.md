# Module 3: Advanced Workflows

Build advanced agentic workflows with the NeMo Agent Toolkit, including RAG pipelines and multi-framework integrations.

## Prerequisites

- Complete Module 2 (NeMo Agent Toolkit)
- NVIDIA API Key set in the root `.env` file

## Setup

```bash
cd module-3-advanced-workflows
uv sync
uv run jupyter lab
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_RAG_It_Is_Ragtime.ipynb` | Build a RAG pipeline using NAT with LlamaIndex, FAISS vector store, and NVIDIA reranking |
| `02_Surge_of_Agents_with_NeMo_Agent_Toolkit.ipynb` | Integrate multiple agent frameworks (OpenAI, LangGraph, CrewAI) into a single NAT workflow |
