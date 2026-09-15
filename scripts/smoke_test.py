"""Run a real artifact through the same interface an application would call."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ncfrec.recommender import NCFRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "artifacts" / "movielens-100k-neumf",
        help="Directory containing model.pt and metadata.json",
    )
    parser.add_argument("--user-id", type=int, help="Raw MovieLens user ID; defaults to the first known user")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = NCFRecommender.from_artifact(args.artifact_dir, device=args.device)
    user_id = args.user_id
    if user_id is None:
        user_id = min(service.metadata.raw_user_to_index)
    health = service.health()
    recommendations = service.recommend(user_id, k=args.k)
    if health.get("status") != "ready":
        raise RuntimeError(f"recommender is not ready: {health}")
    if len(recommendations) != args.k:
        raise RuntimeError(f"expected {args.k} recommendations, received {len(recommendations)}")
    scores = [recommendation.score for recommendation in recommendations]
    if scores != sorted(scores, reverse=True):
        raise RuntimeError("recommendations are not ranked by descending score")
    if len({recommendation.movie_id for recommendation in recommendations}) != len(recommendations):
        raise RuntimeError("recommendations contain duplicate movie IDs")

    payload = {
        "health": health,
        "user_id": user_id,
        "recommendations": [item.to_dict() for item in recommendations],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
