import pandas as pd

from .contracts import validate_candidates_list


REQUIRED_PARTICIPANT_COLUMNS = ["ParticipantKey", "Candidates"]


def validate_participants_candidates_df(participants_df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PARTICIPANT_COLUMNS if c not in participants_df.columns]
    if missing:
        raise ValueError(f"participants_df missing required columns: {missing}")


def explode_candidates(participants_df: pd.DataFrame) -> pd.DataFrame:
    validate_participants_candidates_df(participants_df)

    normalized_rows = []

    for _, row in participants_df[["ParticipantKey", "Candidates"]].iterrows():
        participant_key = row["ParticipantKey"]
        candidates = row["Candidates"]

        if candidates is None:
            candidates = []

        if not isinstance(candidates, list):
            raise ValueError(
                f"Candidates for ParticipantKey={participant_key} must be a list, got {type(candidates).__name__}."
            )

        validate_candidates_list(candidates)

        for candidate in candidates:
            record = {"ParticipantKey": participant_key}
            record.update(candidate)
            normalized_rows.append(record)

    if not normalized_rows:
        return pd.DataFrame(columns=["ParticipantKey"])

    return pd.DataFrame(normalized_rows)
