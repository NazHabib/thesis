import pandas as pd
import json

# Load outputs
fired = pd.read_csv("rules_safety_engine/output/fired_rules.csv")

recs = []
with open("rules_safety_engine/output/recommendations.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        recs.append(json.loads(line))
recs = pd.DataFrame(recs)

conflicts = []
with open("rules_safety_engine/output/conflicts.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        conflicts.append(json.loads(line))
conflicts = pd.DataFrame(conflicts)

# Rule firing frequency
rule_counts = (
    fired.groupby("rule_id")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="n_fired")
)
rule_counts["pct_of_participants"] = rule_counts["n_fired"] / fired["participant_key"].nunique()

# Recommendation frequency
rec_counts = (
    recs.groupby("action_code")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="n_recommendations")
)

# Conflict frequency
if not conflicts.empty and "reason" in conflicts.columns:
    conflict_counts = (
        conflicts.groupby("reason")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="n_conflicts")
    )
else:
    conflict_counts = pd.DataFrame(columns=["reason", "n_conflicts"])

# Per-patient recommendation stats
recs_per_patient = recs.groupby("participant_key").size()

summary = {
    "n_participants_with_recommendations": int(recs["participant_key"].nunique()),
    "avg_recommendations_per_patient": float(recs_per_patient.mean()),
    "median_recommendations_per_patient": float(recs_per_patient.median()),
    "max_recommendations_per_patient": int(recs_per_patient.max()),
    "total_fired_rules": int(len(fired)),
    "total_recommendations": int(len(recs)),
    "total_conflicts": int(len(conflicts)),
}

# Save outputs
rule_counts.to_csv("rules_safety_engine/output/rule_coverage.csv", index=False)
rec_counts.to_csv("rules_safety_engine/output/recommendation_coverage.csv", index=False)
conflict_counts.to_csv("rules_safety_engine/output/conflict_summary.csv", index=False)

with open("rules_safety_engine/output/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Analysis complete")
print(json.dumps(summary, indent=2))
