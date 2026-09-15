"""Data split and negative-sampling contract tests."""

import pandas as pd

from ncfrec.data import prepare_data, sample_training_instances


def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.DataFrame(
        [
            (10, 100, 5.0, 1),
            (10, 101, 4.0, 2),
            (10, 102, 5.0, 3),
            (10, 103, 4.0, 4),
            (20, 100, 5.0, 1),
            (20, 104, 4.0, 2),
            (20, 102, 2.0, 3),
        ],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.DataFrame(
        [(movie_id, f"Movie {movie_id}") for movie_id in range(100, 105)],
        columns=["movie_id", "title"],
    )
    return ratings, movies


def test_prepare_data_uses_latest_positives_for_holdout() -> None:
    ratings, movies = sample_frames()
    prepared = prepare_data(ratings, movies, positive_rating=4.0)
    validation_items = prepared.validation.loc[
        prepared.validation["user_idx"] == prepared.raw_user_to_index[10], "item_idx"
    ]
    test_items = prepared.test.loc[
        prepared.test["user_idx"] == prepared.raw_user_to_index[10], "item_idx"
    ]
    train_items = prepared.train.loc[prepared.train["user_idx"] == prepared.raw_user_to_index[10]]

    assert prepared.num_users == 2
    assert prepared.num_items == 5
    assert validation_items.item() == prepared.raw_movie_to_index[102]
    assert test_items.item() == prepared.raw_movie_to_index[103]
    assert len(train_items) == 2


def test_negative_sampling_is_reproducible_and_never_samples_rated_items() -> None:
    ratings, movies = sample_frames()
    prepared = prepare_data(ratings, movies, positive_rating=4.0)
    first = sample_training_instances(prepared.train, prepared.num_items, prepared.seen_items, 2, seed=7)
    second = sample_training_instances(prepared.train, prepared.num_items, prepared.seen_items, 2, seed=7)

    pd.testing.assert_frame_equal(first, second)
    negatives = first.loc[first["label"] == 0.0]
    assert all(
        int(row.item_idx) not in prepared.seen_items[int(row.user_idx)]
        for row in negatives.itertuples(index=False)
    )
