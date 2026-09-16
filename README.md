# ncfrec

A compact, production-style movie recommendation project using **Neural Collaborative Filtering (NeuMF)** and the MovieLens 100K dataset.

## What it includes

- One end-to-end notebook for exploration, training, ranking evaluation, artifact export, and inference.
- A reusable `src/ncfrec` package so notebook and serving code share the same implementation.
- A load-once inference service that validates IDs, filters rated movies, and returns Top-N results.
- Versioned artifacts: tensor weights in `model.pt`, inspectable mappings in `metadata.json`, and a cross-file SHA-256 consistency check.
- Unit, bounded train-to-serve integration, artifact-contract, and fail-closed command-line smoke tests.

## Architecture

NeuMF combines two recommendation paths:

1. **GMF:** user and movie embeddings interact element-wise.
2. **MLP:** separate embeddings pass through nonlinear dense layers.
3. Their features are concatenated and mapped to an interaction probability.

Ratings of 4 or 5 become positive interactions. Unrated movies are sampled as negatives. Evaluation uses chronological holdouts with full-catalog **Hit Rate@10** and **NDCG@10**.

## Setup

Python 3.11 or newer is required. The committed `uv.lock` is the reproducible environment used for validation. With [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --python 3.12
```

Or with standard Python tooling:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Train in the notebook

Open `notebooks/ncfrec_training.ipynb`, select the project environment, and run all cells. The notebook downloads MovieLens into ignored `data/` storage and exports this ignored artifact directory:

```text
artifacts/movielens-100k-neumf/
├── metadata.json
└── model.pt
```

The epoch default comes from `TrainingConfig` in `src/ncfrec/config.py` (currently 25). To override it for one run, start Jupyter with `NCFREC_EPOCHS=1` in its environment. After editing Python configuration, restart the notebook kernel and run all cells so Python reloads the module.

## Test like an application

Run unit and bounded train-to-serve integration contracts without downloading the dataset:

```bash
uv run pytest
uv run ruff check src scripts tests
```

After running the notebook, exercise the real exported artifact through the production inference facade:

```bash
uv run python scripts/smoke_test.py --user-id 1 --k 10
```

The command fails unless the model is ready and returns exactly `k` unique, score-ordered recommendations, then prints the validated JSON payload. Application code uses the same interface:

```python
from ncfrec.recommender import NCFRecommender

service = NCFRecommender.from_artifact("artifacts/movielens-100k-neumf")
recommendations = service.recommend(user_id=1, k=10)
```

Unknown users currently raise `KeyError`; a real deployment should route them to a popularity- or content-based cold-start strategy.

## Project layout

```text
ncfrec/
├── notebooks/ncfrec_training.ipynb  # complete experiment and export flow
├── scripts/smoke_test.py            # real-artifact operational check
├── src/ncfrec/
│   ├── artifacts.py                 # versioned save/load boundary
│   ├── config.py                    # validated configuration
│   ├── data.py                      # download, split, sampling, Dataset
│   ├── model.py                     # NeuMF architecture
│   ├── recommender.py               # production inference facade
│   └── training.py                  # train/evaluation utilities
└── tests/                            # data, model, artifact, inference contracts
```

## Data source

MovieLens 100K is published by [GroupLens Research](https://grouplens.org/datasets/movielens/100k/). The downloader verifies pinned SHA-256 digests for the archive and extracted training files before use. Review the dataset README and usage terms before redistributing the data. Dataset descriptions here are rephrased for compliance with licensing restrictions.
