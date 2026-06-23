# MIRROR

**M**ulti-agent **I**ntelligent **R**easoning and **R**esolution for **O**ptimization — a Chain-of-Experts framework that decomposes mathematical optimization problems into specialized agent stages: parameter extraction, mathematical modeling, and code generation, with optional reflection.

## Architecture

```
User Problem
    |
    v
+--- Stage 1: Parameter Extraction (ParaExtractor, ModelAdvisor)
+--- Stage 2: Mathematical Modeling (ModelingExpert)
+--- Stage 3: Code Generation (Compiler)
+--- Stage 4: Answer Synthesis (Solver + Python interpreter)
+--- Reflection: Auto-revision on compile error (ModelingExpert + Compiler)
```

## Key Features

- **Chain-of-Experts**: Multi-stage decomposition with specialized expert agents
- **Reflection**: Automatic revision loop on compile/runtime errors
- **Few-shot Prompting**: Built-in support for subtype-specific examples
- **Direct LLM Invocation**: Removed LCEL chain syntax; uses `SystemMessage` + `HumanMessage`

## Important Notes

> **RAG Status**: This version  does **NOT** include RAG retrieval.
> The `--data_path` and `--persist_dir` arguments in `run_rag_exp.py` are
> **reserved for future integration** and have no effect in the current release.
> RAG functionality is planned for a future version.

## Project Structure

```
MIRROR/
├── main.py                 # Chain-of-Experts pipeline entry
├── run_rag_exp.py          # Batch evaluation script
├── requirements.txt        # Python dependencies
├── .env                    # API credentials (create your own, not tracked)
├── .gitignore
├── agent_teams/            # Expert definitions
│   ├── base_expert.py
│   ├── solver.py
│   ├── agent_team_p/       # Parameter extraction
│   ├── agent_team_m/       # Mathematical modeling
│   └── agent_team_c/       # Code generation
├── utils/                  # Shared utilities
└── datasets/               # Evaluation datasets
    ├── ComplexOR.jsonl
    ├── IndustryOR_fix.jsonl
    ├── Mamo_complexLPfix.jsonl
    ├── NL4opt.jsonl
    └── rag_data/few_shot_subtype.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `gurobipy` requires a separate Gurobi license.

### 2. Configure API Credentials

Create a `.env` file:

```
DASHSCOPE_API_KEY="your-api-key-here"
DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 3. Run Evaluation

```bash
python run_rag_exp.py --dataset datasets/ComplexOR.jsonl --model qwen-plus-2025-09-11 --attempts 3
```

### 4. Run Single Problem

```python
from main import causal_agent

answer, output, ref_acc = causal_agent(
    ground_truth=42.0,
    problem="Minimize x^2 + 2x subject to x >= 0",
    problem_name="test_01",
    model_name="qwen-plus-2025-09-11",
    ifRev=True, attempt=3, temperature=0,
    attention="Return only the optimal value.",
    path="./results",
)
```

## Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `datasets/ComplexOR.jsonl` | Evaluation dataset (JSONL) |
| `--model` | `qwen-plus-2025-09-11` | LLM model name |
| `--attempts` | `3` | Max reflection attempts |
| `--enable_revision` | `True` | Enable revision on errors |
| `--temperature` | `0` | Sampling temperature |
| `--data_path` | `None` | [Reserved] RAG file path |
| `--persist_dir` | `None` | [Reserved] ChromaDB directory |

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| `ComplexOR.jsonl` | Complex operations research problems |
| `IndustryOR_fix.jsonl` | Industrial optimization problems |
| `Mamo_complexLPfix.jsonl` | Complex linear programming |
| `NL4opt.jsonl` | Natural language to optimization |

## Version History

| Version | Description |
|---------|-------------|
| v1 | Original LCEL chain-based implementation |
| v2 | Direct LLM invocation. **No RAG**. |
| v3 | Added Python interpreter, agentic RAG, memory management, middleware |

## License

This project is provided for research purposes.
