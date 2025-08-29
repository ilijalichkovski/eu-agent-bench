# EU-Agent-Bench: Evaluating LLM Agent Compliance with EU Legal Norms

This repository contains **EU-Agent-Bench**, a verifiable human-curated benchmark that evaluates the alignment of Large Language Model (LLM) agents with EU legal norms in situations where benign user inputs could lead to unlawful actions.

## Abstract

Large language models (LLMs) are increasingly deployed as agents in various contexts by providing tools at their disposal. However, LLM agents can exhibit unpredictable behaviors, including taking undesirable and/or unsafe actions. In order to measure the latent propensity of LLM agents for taking illegal actions under an EU legislative context, we introduce EU-Agent-Bench, a verifiable human-curated benchmark that evaluates an agent's alignment with EU legal norms in situations where benign user inputs could lead to unlawful actions. Our benchmark spans scenarios across several categories, including data protection, bias/discrimination, and scientific integrity, with each user request allowing for both compliant and non-compliant execution of the requested actions. Comparing the model's function calls against a rubric exhaustively supported by citations of the relevant legislature, we evaluate the legal compliance of frontier LLMs, and furthermore investigate the compliance effect of providing the relevant legislative excerpts in the agent's system prompt along with explicit instructions to comply. We release a public preview set for the research community, while holding out a private test set to prevent data contamination in evaluating upcoming models. We encourage future work extending agentic safety benchmarks to different legal jurisdictions and to multi-turn and multilingual interactions.

## Benchmark Details

EU-Agent-Bench evaluates LLM agent compliance across multiple categories of EU legislation. Each scenario presents a benign user request that could be executed in either compliant or non-compliant ways, allowing us to measure the latent propensity of models to take illegal actions.

## Repository Structure

- **`benchmark_data/`** - Public preview benchmark dataset
- **`benchmark_data_injected/`** - Benchmark with legislative excerpts injected into system prompts
- **`original_data/`** - Original benchmark data before processing
- **`results/`** - Model evaluation results across different LLMs
- **`results_paper/`** - Analysis results and visualizations for the research paper
- **`run_benchmark.py`** - Main script for running benchmark evaluations
- **`data_analysis.py`** - Analysis script for processing and visualizing results
- **`utils.py`** - Utility functions for benchmark processing
- **`formatting.py`** - Data formatting utilities
- **`requirements.txt`** - Python dependencies

## Quick Start

### Prerequisites

We recommend using `uv` for dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

### Running the Benchmark

The main interface for running the benchmark is the `run_bench.sh` script:

```bash
./run_bench.sh
```

This script runs the benchmark across multiple models and datasets in parallel. You can configure the script by editing the variables at the top:

- `AUTOMATIC_COMMIT`: Set to `true` to automatically commit results
- `INJECTED_REGULATIONS`: Set to `true` to run with legislative excerpts in system prompts  
- `OUTPUT_DIR`: Directory where results will be saved
- `models`: Array of models to evaluate
- `datasets`: Array of datasets to test on

The script requires an `OPENROUTER_API_KEY` environment variable to be set. You can create a `.env` file in the project root:

```bash
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

For individual runs, you can also use the underlying Python script directly:

```bash
uv run python run_benchmark.py --model "model_name" --data_path "benchmark_data/dataset.json"
```

### Analyzing Results

To analyze benchmark results and generate visualizations:

```bash
uv run python data_analysis.py
```

## Benchmark Design

### Evaluation Methodology

Each test case in EU-Agent-Bench:

1. **Presents a realistic scenario** where an AI agent has access to specific tools
2. **Provides a benign user request** that doesn't explicitly ask for illegal actions
3. **Creates decision points** where the model must choose between compliant and non-compliant approaches
4. **Enables objective assessment** through comparison against legal rubrics with extensive legislative citations

### Legal Compliance Assessment

Model responses are evaluated against rubrics that are:
- **Exhaustively supported** by citations of relevant EU legislation
- **Verifiable** through direct reference to legal texts
- **Comprehensive** across different categories of legal compliance

### Injection Study

The benchmark includes a study on the compliance effect of:
- Providing relevant legislative excerpts in the agent's system prompt
- Including explicit instructions to comply with EU legal norms

## Model Evaluation Results

The benchmark has been evaluated on several frontier LLMs, with results available in the `results/` directory. Analysis scripts and visualizations can be found in `results_paper/`.

## Data Access

- **Public Preview**: Available in `benchmark_data/` for research community use
- **Private Test Set**: Held out to prevent data contamination in future model evaluations

## Future Work

We encourage extensions of this work to:
- Different legal jurisdictions beyond the EU
- Multi-turn agent interactions
- Multilingual evaluation scenarios
- Additional categories of legal compliance

## Citation

If you use EU-Agent-Bench in your research, please cite:

```
[Citation information to be added]
```

## Contributing

We welcome contributions that extend the benchmark to new legal domains, improve evaluation methodologies, or provide analysis of model performance across different legal frameworks.

## License

This project is licensed under the MIT license.