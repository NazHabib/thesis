from pathlib import Path

import pytest

from rules_safety_engine.rules_engine.utils import validate_ruleset_schema


def test_valid_ruleset_schema():
    payload = {
        "rules": [
            {
                "rule_id": "HTN_CRISIS_ESCALATE",
                "rule_version": "1.0.0",
                "name": "Escalate hypertensive crisis",
                "group": "safety",
                "priority": 1000,
                "condition": "SBP_mean >= 180 or DBP_mean >= 120",
                "missing_policy": "na_false",
                "severity": "critical",
                "action_type": "escalate",
                "action_code": "ER_REFERRAL"
            }
        ]
    }

    schema_path = Path("rules_safety_engine/rules/rule_schema.json")
    validate_ruleset_schema(payload, schema_path)


def test_invalid_ruleset_schema_missing_field():
    payload = {
        "rules": [
            {
                "rule_id": "BAD_RULE",
                "rule_version": "1.0.0",
                "name": "Bad rule",
                "group": "safety",
                "priority": 100,
                "condition": "SBP_mean >= 180",
                "missing_policy": "na_false",
                "severity": "critical",
                "action_type": "escalate"
            }
        ]
    }

    schema_path = Path("rules_safety_engine/rules/rule_schema.json")

    with pytest.raises(ValueError):
        validate_ruleset_schema(payload, schema_path)


def test_invalid_ruleset_schema_bad_enum():
    payload = {
        "rules": [
            {
                "rule_id": "BAD_ENUM",
                "rule_version": "1.0.0",
                "name": "Bad enum rule",
                "group": "safety",
                "priority": 100,
                "condition": "SBP_mean >= 180",
                "missing_policy": "na_something",
                "severity": "critical",
                "action_type": "escalate",
                "action_code": "ER_REFERRAL"
            }
        ]
    }

    schema_path = Path("rules_safety_engine/rules/rule_schema.json")

    with pytest.raises(ValueError):
        validate_ruleset_schema(payload, schema_path)
