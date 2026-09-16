"""Bounded integration checks for notebook syntax and the train-to-serve path."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ncfrec.artifacts import save_artifact
from ncfrec.config import ModelConfig, TrainingConfig
from ncfrec.data import InteractionDataset, prepare_data, sample_training_instances
from ncfrec.model import NeuMF
from ncfrec.training import evaluate_ranking, set_seed, train_one_epoch

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_code_cells_compile() -> None:
    notebook = nbformat.read(ROOT / "notebooks" / "ncfrec_training.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]

    assert code_cells
    notebook_source = "\n".join(code_cells)
    assert "TrainingConfig.from_env(" in notebook_source
    assert 'os.getenv("NCFREC_EPOCHS", "5")' not in notebook_source
    for index, source in enumerate(code_cells):
        compile(source, f"notebook-cell-{index}", "exec")


def test_notebook_training_cell_honors_selected_epoch_count() -> None:
    notebook = nbformat.read(ROOT / "notebooks" / "ncfrec_training.ipynb", as_version=4)
    training_source = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "for epoch in range" in cell.source
    )
    interactions = pd.DataFrame({"user_idx": [0], "item_idx": [0]})
    prepared = SimpleNamespace(train=interactions, validation=interactions)
    calls: list[int] = []

    def fake_train_one_epoch(*_args: object) -> float:
        calls.append(len(calls) + 1)
        return 0.1

    scope = {
        "pd": pd,
        "prepared": prepared,
        "training_config": TrainingConfig(epochs=3),
        "model": object(),
        "training_loader": object(),
        "optimizer": object(),
        "device": torch.device("cpu"),
        "train_one_epoch": fake_train_one_epoch,
        "evaluate_ranking": lambda *_args: {"hit_rate@10": 0.0, "ndcg@10": 0.0},
    }

    exec(training_source, scope)  # noqa: S102 - execute the checked-in notebook cell as an integration test

    assert calls == [1, 2, 3]
    assert scope["history_frame"]["epoch"].tolist() == [1, 2, 3]


def test_bounded_train_export_reload_and_cli_smoke(tmp_path: Path) -> None:
    set_seed(11)
    ratings = pd.DataFrame(
        [
            (10, 100, 5.0, 1),
            (10, 101, 4.0, 2),
            (10, 102, 5.0, 3),
            (10, 103, 4.0, 4),
            (20, 100, 4.0, 1),
            (20, 104, 5.0, 2),
            (20, 105, 4.0, 3),
            (20, 106, 5.0, 4),
        ],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.DataFrame(
        [(movie_id, f"Movie {movie_id}") for movie_id in range(100, 108)],
        columns=["movie_id", "title"],
    )
    prepared = prepare_data(ratings, movies)
    sampled = sample_training_instances(prepared.train, prepared.num_items, prepared.seen_items, 1, seed=11)
    loader = DataLoader(InteractionDataset(sampled), batch_size=4, shuffle=False)
    model = NeuMF(
        ModelConfig(
            num_users=prepared.num_users,
            num_items=prepared.num_items,
            embedding_dim=4,
            mlp_layers=(8, 4),
            dropout=0.0,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    blocked = {
        int(user): {int(item) for item in group["item_idx"]}
        for user, group in prepared.train.groupby("user_idx")
    }
    metrics = evaluate_ranking(model, prepared.validation, blocked, k=3, device=torch.device("cpu"))
    artifact_dir = save_artifact(
        model,
        tmp_path / "artifact",
        prepared.raw_user_to_index,
        prepared.raw_movie_to_index,
        prepared.movie_titles_by_index,
        prepared.seen_items,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "smoke_test.py"),
            "--artifact-dir",
            str(artifact_dir),
            "--user-id",
            "10",
            "--k",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert loss > 0.0
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert payload["health"]["status"] == "ready"
    assert len(payload["recommendations"]) == 2
