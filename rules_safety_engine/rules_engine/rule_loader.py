from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import yaml

from .models import Rule
from .utils import validate_ruleset_schema


def load_rules_from_yaml(rule_path: Path, schema_path: Path | None = None) -> List[Rule]:
    if not rule_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rule_path}")

    with open(rule_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Rules payload must be a dict at top level: {rule_path}")

    if schema_path is not None:
        validate_ruleset_schema(payload, schema_path)

    rules_raw = payload.get("rules", [])
    if not isinstance(rules_raw, list):
        raise ValueError(f"'rules' must be a list in: {rule_path}")

    rules: List[Rule] = []
    for item in rules_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Each rule must be an object in: {rule_path}")

        rule = Rule(**item)
        rule.validate()
        rules.append(rule)

    return rules


def load_rules_from_many_yaml(rule_paths: Iterable[Path], schema_path: Path | None = None) -> List[Rule]:
    out: List[Rule] = []
    for path in rule_paths:
        out.extend(load_rules_from_yaml(path, schema_path=schema_path))
    return out
