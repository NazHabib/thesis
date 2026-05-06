from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("rules_safety_engine/output")

FIRED_PATH = OUTPUT_DIR / "fired_rules.csv"
RECS_PATH = OUTPUT_DIR / "recommendations.jsonl"
CONFLICTS_PATH = OUTPUT_DIR / "conflicts.jsonl"


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required rules-engine output not found: {path}. "
            "Run the rules engine before running coverage analysis."
        )


def _load_jsonl(path: Path) -> pd.DataFrame:
    _require_file(path)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return pd.DataFrame(rows)


def main() -> None:
    _require_file(FIRED_PATH)

    fired = pd.read_csv(FIRED_PATH)
    recs = _load_jsonl(RECS_PATH)
    conflicts = _load_jsonl(CONFLICTS_PATH)

    if fired.empty:
        raise ValueError("fired_rules.csv is empty. Cannot compute rule coverage.")

    if "participant_key" not in fired.columns:
        raise ValueError("fired_rules.csv must include participant_key.")

    if "rule_id" not in fired.columns:
        raise ValueError("fired_rules.csv must include rule_id.")

    if recs.empty:
        recs = pd.DataFrame(columns=["participant_key", "action_code"])

    if conflicts.empty:
        conflicts = pd.DataFrame(columns=["reason"])

    participant_count = fired["participant_key"].nunique()

    rule_counts = (
        fired.groupby("rule_id")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="n_fired")
    )
    rule_counts["pct_of_participants"] = (
        rule_counts["n_fired"] / participant_count if participant_count else 0.0
    )

    if "action_code" in recs.columns and not recs.empty:
        rec_counts = (
            recs.groupby("action_code")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="n_recommendations")
        )
    else:
        rec_counts = pd.DataFrame(columns=["action_code", "n_recommendations"])

    if "reason" in conflicts.columns and not conflicts.empty:
        conflict_counts = (
            conflicts.groupby("reason")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="n_conflicts")
        )
    else:
        conflict_counts = pd.DataFrame(columns=["reason", "n_conflicts"])

    if "participant_key" in recs.columns and not recs.empty:
        recs_per_patient = recs.groupby("participant_key").size()
        avg_recs = float(recs_per_patient.mean())
        median_recs = float(recs_per_patient.median())
        max_recs = int(recs_per_patient.max())
        n_participants_with_recs = int(recs["participant_key"].nunique())
    else:
        avg_recs = 0.0
        median_recs = 0.0
        max_recs = 0
        n_participants_with_recs = 0

    summary = {
        "n_participants_with_recommendations": n_participants_with_recs,
        "avg_recommendations_per_patient": avg_recs,
        "median_recommendations_per_patient": median_recs,
        "max_recommendations_per_patient": max_recs,
        "total_fired_rules": int(len(fired)),
        "total_recommendations": int(len(recs)),
        "total_conflicts": int(len(conflicts)),
    }

    rule_counts.to_csv(OUTPUT_DIR / "rule_coverage.csv", index=False)
    rec_counts.to_csv(OUTPUT_DIR / "recommendation_coverage.csv", index=False)
    conflict_counts.to_csv(OUTPUT_DIR / "conflict_summary.csv", index=False)

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Analysis complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
