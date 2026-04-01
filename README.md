# llm-cost-estimator

A command-line tool that tells you how many tokens your text uses and what it will cost across different OpenAI models — before you make any API calls.

Built with tiktoken, which is the same tokenizer OpenAI uses internally, so the counts are exact and not estimates.

---

## Why I built this

When you're working with LLM APIs, you don't really know what something costs until after you've already sent the request. This tool lets you check token counts and costs locally, with no API key required, so you can make informed decisions about which model to use and how to structure your prompts.

---

## Installation

You'll need Python 3.9 or higher.

```bash
git clone https://github.com/yourusername/llm-cost-estimator.git
cd llm-cost-estimator
python -m venv venv
```

On Mac/Linux:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

Then install:
```bash
pip install -e .
```

---

## Usage

Estimate a string directly:
```bash
python -m llm_cost.cli estimate "Explain how neural networks work"
```

Estimate from a file:
```bash
python -m llm_cost.cli estimate --file prompt.txt
```

Pipe from stdin:
```bash
cat document.txt | python -m llm_cost.cli estimate
```

Filter to specific models:
```bash
python -m llm_cost.cli estimate --file prompt.txt --models gpt-4o,gpt-3.5-turbo
```

See how your text is split into tokens:
```bash
python -m llm_cost.cli estimate "Your text here" --show-tokens
```

Get JSON output for scripting:
```bash
python -m llm_cost.cli estimate --file prompt.txt --format json
```

List all available models and their pricing:
```bash
python -m llm_cost.cli models
```

---

## What the output means

**Tokens** — the number of tokens in your input text. LLM APIs charge per token, not per character or word.

**Input cost** — what it costs to send your text as a prompt to that model.

**Output / 1k tokens** — what it costs for the model to generate 1000 tokens in response.

**Context used %** — how much of the model's context window your text occupies. If this gets too high (above 75%), the model won't have enough room to generate a meaningful response.

---

## How it works

tiktoken implements Byte Pair Encoding (BPE), which is how OpenAI tokenizes text under the hood. The tokenizer starts with individual characters and iteratively merges the most frequent pairs until it has a vocabulary of roughly 100,000 subword tokens. Common English words tend to be a single token, while rare words and non-Latin scripts get split into multiple tokens.

Different models use different encodings. GPT-4o uses `o200k_base` while GPT-3.5 and GPT-4 use `cl100k_base`, which is why token counts can vary slightly between models for the same input.

---

## Updating prices

Prices are stored in `config/pricing.yaml`. If OpenAI changes their pricing, you can update the numbers there without touching any code.

```yaml
models:
  gpt-4o:
    encoding: o200k_base
    input: 2.50       # USD per 1 million tokens
    output: 10.00
    context_window: 128000
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project structure

```
llm-cost-estimator/
├── llm_cost/
│   ├── tokenizer.py     # tiktoken wrapper
│   ├── pricing.py       # cost calculation
│   ├── display.py       # terminal output
│   └── cli.py           # CLI commands
├── config/
│   └── pricing.yaml     # model pricing
├── tests/
│   └── test_tokenizer.py
└── pyproject.toml
```

---

## Requirements

- Python 3.9+
- tiktoken
- typer
- rich
- pyyaml
