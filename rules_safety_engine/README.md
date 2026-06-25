# Rules Safety Engine

Purpose:
Deterministic clinical safety and recommendation rules for SafeHybridCare.

Inputs:
Engineered participant health context with NHANES-style columns.

Outputs:
- fired_rules.csv
- recommendations.jsonl
- conflicts.jsonl
- manifest.json
- warnings.json

Safety design:
- no LLM explanations
- no prescribing
- acute escalation rules outrank lifestyle recommendations
- explicit conflict and suppression handling
- guideline citations and provenance per rule

Commands:
python -m rules_safety_engine.rules_engine.cli validate --rules rules_safety_engine/rules/example_rules.yaml
python -m rules_safety_engine.rules_engine.cli run --rules rules_safety_engine/rules/example_rules.yaml --data nhanes_step1_out/NHANES_Master_engineered.parquet --out rules_safety_engine/output
python manage.py validate_rulebase
python manage.py export_rule_inventory
python manage.py export_rule_coverage_report
