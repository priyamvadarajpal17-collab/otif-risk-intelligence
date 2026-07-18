# Teammate Setup Guide

This guide takes a new laptop from an empty directory to the running OTIF Risk
Intelligence control tower.

## 1. Install prerequisites

The project uses Python 3.12 through
[`uv`](https://docs.astral.sh/uv/) and does not require a manually created
virtual environment.

### macOS

```bash
brew install uv libomp
```

`libomp` is required by XGBoost on macOS.

### Windows

```powershell
winget install --id astral-sh.uv -e
```

### Linux

Install `uv` using your approved package manager or:

```bash
pipx install uv
```

On Debian/Ubuntu, install the OpenMP runtime if XGBoost reports that it is
missing:

```bash
sudo apt-get install libgomp1
```

Verify the installation:

```bash
uv --version
git --version
```

## 2. Clone the repository

```bash
git clone https://github.com/priyamvadarajpal17-collab/otif-risk-intelligence.git
cd otif-risk-intelligence
```

## 3. Install project dependencies

```bash
uv sync
```

`uv` reads `.python-version`, installs Python 3.12 when needed, creates
`.venv`, and installs the locked dependencies from `uv.lock`.

## 4. Generate the main demo artifacts

```bash
uv run otif-risk \
  --orders 2500 \
  --seed 42 \
  --output-dir artifacts
```

This command:

1. generates the synthetic supply-chain data;
2. validates the tables;
3. builds point-in-time features;
4. trains XGBoost and the Bayesian mechanism network;
5. scores held-out orders;
6. creates explanations, recommendations, rollups, models and manifests.

## 5. Start the control tower

```bash
uv run streamlit run src/otif_risk/app.py
```

Open:

```text
http://localhost:8501
```

The AI Copilot works without an API key by using its grounded deterministic
fallback.

## 6. Optional: enable live OpenAI Copilot

Never paste or commit an API key. Create a local ignored `.env`:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` and set:

```env
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-5-mini
OTIF_LLM_MODE=auto
```

Restart Streamlit after saving. In `auto` mode the Copilot uses OpenAI when
available and falls back automatically if the API is unavailable or a response
fails grounding validation.

## 7. Generate the complete competition demo

Run these commands in order.

### Prediction benchmark

```bash
uv run otif-benchmark \
  --seeds 1 2 3 4 5 \
  --orders 2500 \
  --output-dir artifacts \
  --benchmark-path artifacts/benchmark.json
```

### Decision-policy value benchmark

```bash
uv run otif-policy-benchmark \
  --seeds 1 2 3 4 5 \
  --orders 2500 \
  --benchmark-path artifacts/policy_benchmark.json
```

### Governed 90-day operations replay

```bash
uv run otif-ops \
  --orders 2500 \
  --seed 42 \
  --replay-days 90 \
  --output-dir artifacts \
  --policy-value-reference-path artifacts/policy_benchmark.json
```

Restart Streamlit after the commands finish. The additional artifacts enable
the Operations, Policy Value and Governance views.

The five-seed benchmarks and 90-day replay take substantially longer than the
quick-start pipeline.

## 8. Open the interactive system walkthrough

In a second terminal:

```bash
uv run python -m http.server 8731 --directory docs
```

Open:

```text
http://localhost:8731/system-walkthrough.html
```

The walkthrough explains the complete architecture with interactive diagrams
and measured results.

## 9. Useful operating modes

### Fast local iteration

```bash
uv run otif-risk --orders 1000 --seed 42 --output-dir artifacts
```

### Standard demo

```bash
uv run otif-risk --orders 2500 --seed 42 --output-dir artifacts
```

### Larger final run

```bash
uv run otif-risk --orders 5000 --seed 42 --output-dir artifacts
```

Use 2,500 orders during development. A 5,000-order run provides more stable
history and mechanism estimates but takes longer, especially for multi-seed
policy evaluation.

## Troubleshooting

### XGBoost or OpenMP error on macOS

```bash
brew install libomp
uv sync
```

Then rerun the pipeline.

### Port 8501 is already in use

Start Streamlit on another port:

```bash
uv run streamlit run src/otif_risk/app.py --server.port 8502
```

### UI says artifacts are not ready

Run the main pipeline first:

```bash
uv run otif-risk --orders 2500 --seed 42 --output-dir artifacts
```

Then refresh the browser.

### AI Copilot shows fallback mode

Fallback mode is fully functional. For live mode, confirm that `.env` contains
`OPENAI_API_KEY`, `OPENAI_MODEL=gpt-5-mini`, and `OTIF_LLM_MODE=auto`, then
restart Streamlit.

### Reset generated data

Artifacts are generated and gitignored. To start clean, move or delete the
local `artifacts/` directory, then rerun the commands above. Do not delete
source files or `uv.lock`.
