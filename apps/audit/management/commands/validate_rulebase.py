from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from rules_safety_engine.rules_engine.rule_loader import load_rules_from_yaml
from rules_safety_engine.rules_engine.rulebase_audit import audit_rulebase


class Command(BaseCommand):
    help = "Validate the rules safety engine rulebase and export audit issues."

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
        parser.add_argument(
            "--fail-on-warning",
            action="store_true",
        )

    def handle(self, *args, **options):
        rules_path = Path(options["rules"])
        schema_path = Path(options["schema"])
        output_dir = Path(options["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Loading and validating rules...")
        rules = load_rules_from_yaml(rules_path, schema_path=schema_path)

        self.stdout.write("Running rulebase audit...")
        audit = audit_rulebase(rules)

        issues_df = pd.DataFrame(audit["issues"])
        issues_df.to_csv(output_dir / "rulebase_issues.csv", index=False)

        with open(output_dir / "rulebase_validation.json", "w", encoding="utf-8") as f:
            json.dump(audit["summary"], f, indent=2, ensure_ascii=False)

        summary = audit["summary"]

        self.stdout.write(self.style.SUCCESS("Rulebase validation complete."))
        self.stdout.write(f"Rules total: {summary['n_rules_total']}")
        self.stdout.write(f"Errors: {summary['n_error_issues']}")
        self.stdout.write(f"Warnings: {summary['n_warning_issues']}")
        self.stdout.write(f"Saved issues to: {output_dir / 'rulebase_issues.csv'}")
        self.stdout.write(f"Saved summary to: {output_dir / 'rulebase_validation.json'}")

        if summary["n_error_issues"] > 0:
            raise CommandError("Rulebase validation failed with error-level issues.")

        if options["fail_on_warning"] and summary["n_warning_issues"] > 0:
            raise CommandError("Rulebase validation failed because warnings were treated as errors.")
