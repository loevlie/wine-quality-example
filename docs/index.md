# Wine Quality

Predict red-wine quality from physicochemical measurements — a complete walkthrough of ml-research-template.

## Quick Start

```bash
# Install
uv sync --extra dev

# Train with defaults
python src/wine_quality/train.py

# Train with overrides
python src/wine_quality/train.py model.lr=1e-3 data.batch_size=128

# Run a named experiment
python src/wine_quality/train.py experiment=example

# Multi-seed run
bash scripts/run_seeds.sh experiment=example seeds="42,123,456,789,1337"
```

## Project Structure

See the README for the full directory layout and tool descriptions.
