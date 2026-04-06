import pandas as pd


def explode_candidates(participants_df: pd.DataFrame) -> pd.DataFrame:
    exploded = participants_df[['ParticipantKey', 'Candidates']].explode('Candidates', ignore_index=True)
    payload = pd.json_normalize(exploded['Candidates'])
    out = pd.concat([exploded[['ParticipantKey']], payload], axis=1)
    return out
