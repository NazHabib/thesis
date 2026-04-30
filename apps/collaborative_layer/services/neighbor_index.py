import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_neighbors(
    participant_keys: list[str],
    matrix,
    k: int = 20,
    similarity_threshold: float = 0.60,
) -> pd.DataFrame:
    if len(participant_keys) == 0:
        return pd.DataFrame(columns=[
            "ParticipantKey",
            "neighbor_participant_key",
            "neighbor_rank",
            "similarity",
        ])

    n_neighbors = min(k + 1, len(participant_keys))

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(matrix)

    distances, indices = nn.kneighbors(matrix)

    rows = []

    for i, participant_key in enumerate(participant_keys):
        rank = 0

        for dist, idx in zip(distances[i], indices[i]):
            neighbor_key = participant_keys[idx]

            if str(neighbor_key) == str(participant_key):
                continue

            similarity = float(1.0 - dist)

            if similarity < similarity_threshold:
                continue

            rank += 1

            rows.append({
                "ParticipantKey": participant_key,
                "neighbor_participant_key": neighbor_key,
                "neighbor_rank": rank,
                "similarity": similarity,
            })

            if rank >= k:
                break

    return pd.DataFrame(rows)
