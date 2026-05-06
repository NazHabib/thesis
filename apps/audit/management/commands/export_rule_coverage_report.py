from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from rules_safety_engine.rules_engine.rule_loader import load_rules_from_yaml
from rules_safety_engine.rules_engine.rulebase_audit import audit_rulebase


class Command(BaseCommand):
    help = "Export an extended rule coverage report for audit and dissertation use."

    def add_arguments(self, parser):
        parser.add_argument(
            "--rules",
            type=str,
            default="rules_safety_engine/rules/example_rules.yaml",
        )
        parser.add_argument(
            "--schema",
            type=str,
            default="rules_safety_engine/rules/rule_schema.json",
        )
        parser.add_argument(
            "--rule-coverage-csv",
            type=str,
            default="rules_safety_engine/output/rule_coverage.csv",
        )
        parser.add_argument(
            "--recommendation-coverage-csv",
            type=str,
            default="rules_safety_engine/output/recommendation_coverage.csv",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="rules_safety_engine/output",
        )

    def handle(self, *args, **options):
        rules_path = Path(options["rules"])
        schema_path = Path(options["schema"])
        rule_coverage_path = Path(options["rule_coverage_csv"])
        recommendation_coverage_path = Path(options["recommendation_coverage_csv"])
        output_dir = Path(options["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Loading rules...")
        rules = load_rules_from_yaml(rules_path, schema_path=schema_path)
        audit = audit_rulebase(rules)

        self.stdout.write("Loading coverage outputs...")
        rule_cov_df = pd.read_csv(rule_coverage_path) if rule_coverage_path.exists() else pd.DataFrame()
        rec_cov_df = pd.read_csv(recommendation_coverage_path) if recommendation_coverage_path.exists() else pd.DataFrame()

        inventory_df = pd.DataFrame(audit["inventory_rows"])

        coverage_by_group_df = (
            inventory_df.groupby("group", as_index=False)
            .agg(
                n_rules=("rule_id", "count"),
                n_enabled=("enabled", "sum"),
                mean_priority=("priority", "mean"),
                n_escalate=("action_type", lambda s: int((s == "escalate").sum())),
                n_recommend=("action_type", lambda s: int((s == "recommend").sum())),
                n_inform=("action_type", lambda s: int((s == "inform").sum())),
                n_critical=("severity", lambda s: int((s == "critical").sum())),
                n_high=("severity", lambda s: int((s == "high").sum())),
            )
            .sort_values(["n_rules", "mean_priority"], ascending=[False, False], kind="stable")
        )

        coverage_by_action_type_df = (
            inventory_df.groupby("action_type", as_index=False)
            .agg(
                n_rules=("rule_id", "count"),
                mean_priority=("priority", "mean"),
            )
            .sort_values("n_rules", ascending=False, kind="stable")
        )

        coverage_by_severity_df = (
            inventory_df.groupby("severity", as_index=False)
            .agg(
                n_rules=("rule_id", "count"),
                mean_priority=("priority", "mean"),
            )
            .sort_values("n_rules", ascending=False, kind="stable")
        )

        suppression_profile_df = inventory_df[
            ["rule_id", "group", "action_type", "severity", "n_suppresses", "suppresses"]
        ].sort_values(["n_suppresses", "severity"], ascending=[False, True], kind="stable")

        citation_profile_df = inventory_df[
            ["rule_id", "group", "action_code", "n_citations", "has_explanation_template"]
        ].sort_values(["n_citations", "rule_id"], ascending=[False, True], kind="stable")

        if not rule_cov_df.empty:
            firing_join_df = inventory_df.merge(
                rule_cov_df,
                on="rule_id",
                how="left",
            )
            firing_join_df["n_fired"] = firing_join_df["n_fired"].fillna(0).astype(int)
            firing_join_df["pct_of_participants"] = firing_join_df["pct_of_participants"].fillna(0.0)
        else:
            firing_join_df = inventory_df.copy()
            firing_join_df["n_fired"] = 0
            firing_join_df["pct_of_participants"] = 0.0

        if not rec_cov_df.empty:
            recommendation_join_df = inventory_df.merge(
                rec_cov_df,
                on="action_code",
                how="left",
            )
            recommendation_join_df["n_recommendations"] = recommendation_join_df["n_recommendations"].fillna(0).astype(int)
        else:
            recommendation_join_df = inventory_df.copy()
            recommendation_join_df["n_recommendations"] = 0

        underrepresented_groups = coverage_by_group_df.loc[
            coverage_by_group_df["n_rules"] <= 2,
            ["group", "n_rules"]
        ].to_dict(orient="records")

        low_frequency_active_rules = firing_join_df.loc[
            firing_join_df["enabled"].astype(bool) & (firing_join_df["n_fired"] <= 10),
            ["rule_id", "group", "action_code", "severity", "n_fired"]
        ].sort_values(["n_fired", "rule_id"], ascending=[True, True], kind="stable")

        high_frequency_rules = firing_join_df.loc[
            firing_join_df["n_fired"] > 0,
            ["rule_id", "group", "action_code", "severity", "n_fired", "pct_of_participants"]
        ].sort_values(["n_fired", "pct_of_participants"], ascending=[False, False], kind="stable")

        summary = {
            "n_rules_total": int(len(inventory_df)),
            "n_groups_total": int(inventory_df["group"].nunique()),
            "n_action_codes_total": int(inventory_df["action_code"].nunique()),
            "n_groups_underrepresented": int(len(underrepresented_groups)),
            "underrepresented_groups": underrepresented_groups,
            "top_10_rule_groups_by_rule_count": coverage_by_group_df.head(10).to_dict(orient="records"),
            "top_10_high_frequency_rules": high_frequency_rules.head(10).to_dict(orient="records"),
            "top_10_low_frequency_active_rules": low_frequency_active_rules.head(10).to_dict(orient="records"),
            "coverage_observations": [
                "Rule coverage is broad across safety, hypertension, diabetes, obesity, diet, activity, smoking, alcohol, lipids, and multimorbidity.",
                "The current rulebase is strongest in diet, blood pressure, glycaemic, and lifestyle-support recommendations.",
                "Rare escalation rules appear appropriately sparse, reflecting low expected prevalence rather than missing coverage.",
                "The main remaining gap is temporal/longitudinal rule logic rather than basic domain-family absence.",
            ],
        }

        coverage_by_group_df.to_csv(output_dir / "coverage_by_group.csv", index=False)
        coverage_by_action_type_df.to_csv(output_dir / "coverage_by_action_type.csv", index=False)
        coverage_by_severity_df.to_csv(output_dir / "coverage_by_severity.csv", index=False)
        suppression_profile_df.to_csv(output_dir / "suppression_profile.csv", index=False)
        citation_profile_df.to_csv(output_dir / "citation_profile.csv", index=False)
        firing_join_df.to_csv(output_dir / "rule_inventory_with_firing.csv", index=False)
        recommendation_join_df.to_csv(output_dir / "rule_inventory_with_recommendations.csv", index=False)
        low_frequency_active_rules.to_csv(output_dir / "low_frequency_active_rules.csv", index=False)
        high_frequency_rules.to_csv(output_dir / "high_frequency_rules.csv", index=False)

        with open(output_dir / "rule_coverage_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS("Rule coverage report exported."))
        self.stdout.write(f"Saved to: {output_dir}")
