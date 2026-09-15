"""MovieLens 100K download, preprocessing, splitting, and sampling."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_100K_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
RATINGS_SHA256 = "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490"
MOVIES_SHA256 = "553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701"


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"{label} checksum mismatch: {path}")


@dataclass
class PreparedData:
    """Encoded interactions and metadata needed for training and serving."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    raw_user_to_index: dict[int, int]
    raw_movie_to_index: dict[int, int]
    movie_titles_by_index: list[str]
    seen_items: dict[int, set[int]]

    @property
    def num_users(self) -> int:
        return len(self.raw_user_to_index)

    @property
    def num_items(self) -> int:
        return len(self.raw_movie_to_index)


class InteractionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Torch dataset of encoded user/item pairs and binary labels."""

    def __init__(self, frame: pd.DataFrame) -> None:
        required = {"user_idx", "item_idx", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"frame must contain columns: {sorted(required)}")
        self.users = torch.as_tensor(frame["user_idx"].to_numpy(), dtype=torch.long)
        self.items = torch.as_tensor(frame["item_idx"].to_numpy(), dtype=torch.long)
        self.labels = torch.as_tensor(frame["label"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[index], self.items[index], self.labels[index]


def download_movielens(data_dir: str | Path) -> Path:
    """Download and extract only the MovieLens files used by this project."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "ml-100k"
    ratings_path = raw_dir / "u.data"
    movies_path = raw_dir / "u.item"
    if ratings_path.exists() and movies_path.exists():
        _verify_file(ratings_path, RATINGS_SHA256, "MovieLens ratings")
        _verify_file(movies_path, MOVIES_SHA256, "MovieLens movies")
        return raw_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "ml-100k.zip"
    temporary_archive = data_dir / "ml-100k.zip.part"
    if not archive_path.exists():
        urllib.request.urlretrieve(MOVIELENS_100K_URL, temporary_archive)
        _verify_file(temporary_archive, MOVIELENS_100K_SHA256, "MovieLens archive")
        temporary_archive.replace(archive_path)
    else:
        _verify_file(archive_path, MOVIELENS_100K_SHA256, "MovieLens archive")

    raw_dir.mkdir(parents=True, exist_ok=True)
    members = {"ml-100k/u.data": ratings_path, "ml-100k/u.item": movies_path}
    with zipfile.ZipFile(archive_path) as archive:
        for member_name, destination in members.items():
            with archive.open(member_name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    _verify_file(ratings_path, RATINGS_SHA256, "MovieLens ratings")
    _verify_file(movies_path, MOVIES_SHA256, "MovieLens movies")
    return raw_dir


def load_movielens(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens ratings and movie titles into tidy data frames."""
    raw_dir = Path(raw_dir)
    ratings = pd.read_csv(
        raw_dir / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": "int64", "movie_id": "int64", "rating": "float32", "timestamp": "int64"},
    )
    movies = pd.read_csv(
        raw_dir / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"],
        dtype={"movie_id": "int64", "title": "string"},
    )
    return ratings, movies


def prepare_data(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    positive_rating: float = 4.0,
) -> PreparedData:
    """Encode IDs and make per-user chronological train/validation/test splits.

    Users with at least three positive interactions contribute their two latest
    positives to validation and test. Users with fewer positives remain entirely
    in training, avoiding the loss of scarce training signal.
    """
    required_ratings = {"user_id", "movie_id", "rating", "timestamp"}
    if not required_ratings.issubset(ratings.columns):
        raise ValueError(f"ratings must contain columns: {sorted(required_ratings)}")
    if not {"movie_id", "title"}.issubset(movies.columns):
        raise ValueError("movies must contain movie_id and title columns")

    raw_users = sorted(int(value) for value in ratings["user_id"].unique())
    raw_movies = sorted(int(value) for value in ratings["movie_id"].unique())
    raw_user_to_index = {raw_id: index for index, raw_id in enumerate(raw_users)}
    raw_movie_to_index = {raw_id: index for index, raw_id in enumerate(raw_movies)}

    encoded = ratings.copy()
    encoded["user_idx"] = encoded["user_id"].map(raw_user_to_index).astype("int64")
    encoded["item_idx"] = encoded["movie_id"].map(raw_movie_to_index).astype("int64")

    title_by_raw_id = movies.set_index("movie_id")["title"].astype(str).to_dict()
    movie_titles_by_index = [title_by_raw_id.get(movie_id, f"Movie {movie_id}") for movie_id in raw_movies]

    seen_items = {
        int(user_idx): {int(item_idx) for item_idx in group["item_idx"]}
        for user_idx, group in encoded.groupby("user_idx", sort=False)
    }

    positives = encoded.loc[encoded["rating"] >= positive_rating].copy()
    positives = positives.sort_values(["user_idx", "timestamp", "item_idx"]).reset_index(drop=True)
    split = np.full(len(positives), "train", dtype=object)
    for indices in positives.groupby("user_idx", sort=False).groups.values():
        ordered_indices = list(indices)
        if len(ordered_indices) >= 3:
            split[ordered_indices[-2]] = "validation"
            split[ordered_indices[-1]] = "test"
    positives["split"] = split
    columns = ["user_idx", "item_idx", "timestamp"]

    return PreparedData(
        train=positives.loc[positives["split"] == "train", columns].reset_index(drop=True),
        validation=positives.loc[positives["split"] == "validation", columns].reset_index(drop=True),
        test=positives.loc[positives["split"] == "test", columns].reset_index(drop=True),
        raw_user_to_index=raw_user_to_index,
        raw_movie_to_index=raw_movie_to_index,
        movie_titles_by_index=movie_titles_by_index,
        seen_items=seen_items,
    )


def sample_training_instances(
    positives: pd.DataFrame,
    num_items: int,
    excluded_items: Mapping[int, set[int]],
    negatives_per_positive: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Pair each positive with sampled movies the user has never rated."""
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1")
    rng = np.random.default_rng(seed)
    all_items = np.arange(num_items, dtype=np.int64)
    records: list[tuple[int, int, float]] = []
    available_cache: dict[int, np.ndarray] = {}

    for row in positives.itertuples(index=False):
        user_idx = int(row.user_idx)
        item_idx = int(row.item_idx)
        records.append((user_idx, item_idx, 1.0))
        if user_idx not in available_cache:
            blocked = np.fromiter(excluded_items.get(user_idx, set()), dtype=np.int64)
            available_cache[user_idx] = np.setdiff1d(all_items, blocked, assume_unique=False)
        available = available_cache[user_idx]
        if len(available) == 0:
            raise ValueError(f"user {user_idx} has no unrated items to sample")
        sampled = rng.choice(available, size=negatives_per_positive, replace=len(available) < negatives_per_positive)
        records.extend((user_idx, int(negative), 0.0) for negative in sampled)

    frame = pd.DataFrame(records, columns=["user_idx", "item_idx", "label"])
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
