from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from rules_safety_engine.rules_engine.rule_loader import load_rules_from_yaml
from rules_safety_engine.rules_engine.rulebase_audit import audit_rulebase


class Command(BaseCommand):
    help = "Export a rule inventory and summary for the rules safety engine."

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
            "--output-dir",
            type=str,
            default="rules_safety_engine/output",
        )

    def handle(self, *args, **options):
        rules_path = Path(options["rules"])
        schema_path = Path(options["schema"])
        output_dir = Path(options["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Loading rules...")
        rules = load_rules_from_yaml(rules_path, schema_path=schema_path)

        self.stdout.write("Building rule inventory...")
        audit = audit_rulebase(rules)

        inventory_df = pd.DataFrame(audit["inventory_rows"])
        inventory_df.to_csv(output_dir / "rule_inventory.csv", index=False)

        with open(output_dir / "rule_inventory_summary.json", "w", encoding="utf-8") as f:
            json.dump(audit["summary"], f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS("Rule inventory exported."))
        self.stdout.write(f"Saved inventory to: {output_dir / 'rule_inventory.csv'}")
        self.stdout.write(f"Saved summary to: {output_dir / 'rule_inventory_summary.json'}")
