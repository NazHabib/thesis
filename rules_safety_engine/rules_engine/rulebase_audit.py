from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Sequence

from .models import Rule


def build_rule_inventory_rows(rules: Sequence[Rule]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for rule in rules:
        rows.append({
            "rule_id": rule.rule_id,
            "rule_version": rule.rule_version,
            "name": rule.name,
            "group": rule.group,
            "priority": rule.priority,
            "severity": rule.severity,
            "action_type": rule.action_type,
            "action_code": rule.action_code,
            "enabled": rule.enabled,
            "missing_policy": rule.missing_policy,
            "conflict_group": rule.conflict_group,
            "conflict_policy": rule.conflict_policy,
            "n_tags": len(rule.tags or []),
            "tags": "|".join(rule.tags or []),
            "n_suppresses": len(rule.suppresses or []),
            "suppresses": "|".join(rule.suppresses or []),
            "n_citations": len(rule.citations or []),
            "has_explanation_template": bool(str(rule.explanation_template or "").strip()),
            "created_by": rule.created_by,
            "created_at": rule.created_at,
        })

    return rows


def audit_rulebase(rules: Sequence[Rule]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    rule_ids = [r.rule_id for r in rules]
    rule_id_versions = [(r.rule_id, r.rule_version) for r in rules]
    action_codes = [r.action_code for r in rules]

    by_rule_id = Counter(rule_ids)
    by_rule_id_version = Counter(rule_id_versions)
    by_action_code = Counter(action_codes)

    known_rule_ids = {r.rule_id for r in rules}
    known_action_codes = {r.action_code for r in rules}

    for rule_id, count in by_rule_id.items():
        if count > 1:
            issues.append({
                "issue_type": "duplicate_rule_id",
                "severity": "error",
                "rule_id": rule_id,
                "detail": f"rule_id appears {count} times",
            })

    for (rule_id, rule_version), count in by_rule_id_version.items():
        if count > 1:
            issues.append({
                "issue_type": "duplicate_rule_id_version",
                "severity": "error",
                "rule_id": rule_id,
                "rule_version": rule_version,
                "detail": f"(rule_id, rule_version) appears {count} times",
            })

    for action_code, count in by_action_code.items():
        if count > 1:
            issues.append({
                "issue_type": "duplicate_action_code",
                "severity": "warning",
                "action_code": action_code,
                "detail": f"action_code appears {count} times",
            })

    for rule in rules:
        if not str(rule.explanation_template or "").strip():
            issues.append({
                "issue_type": "missing_explanation_template",
                "severity": "warning",
                "rule_id": rule.rule_id,
                "detail": "explanation_template is empty",
            })

        if not rule.citations:
            issues.append({
                "issue_type": "missing_citations",
                "severity": "warning",
                "rule_id": rule.rule_id,
                "detail": "citations list is empty",
            })

        for target in rule.suppresses or []:
            if target not in known_action_codes and target not in known_rule_ids:
                issues.append({
                    "issue_type": "unknown_suppression_target",
                    "severity": "warning",
                    "rule_id": rule.rule_id,
                    "detail": f"suppresses target not found in rule_ids or action_codes: {target}",
                })

        if rule.action_type == "escalate" and rule.severity not in {"high", "critical"}:
            issues.append({
                "issue_type": "weak_escalation_severity",
                "severity": "warning",
                "rule_id": rule.rule_id,
                "detail": f"escalate rule has severity={rule.severity}",
            })

        if rule.conflict_group and not rule.conflict_policy:
            issues.append({
                "issue_type": "missing_conflict_policy",
                "severity": "warning",
                "rule_id": rule.rule_id,
                "detail": "conflict_group is set but conflict_policy is missing",
            })

    summary = {
        "n_rules_total": len(rules),
        "n_rules_enabled": int(sum(1 for r in rules if r.enabled)),
        "n_rules_disabled": int(sum(1 for r in rules if not r.enabled)),
        "counts_by_group": dict(Counter(r.group for r in rules)),
        "counts_by_severity": dict(Counter(r.severity for r in rules)),
        "counts_by_action_type": dict(Counter(r.action_type for r in rules)),
        "counts_by_missing_policy": dict(Counter(r.missing_policy for r in rules)),
        "counts_by_conflict_policy": dict(Counter(str(r.conflict_policy) for r in rules)),
        "n_issues_total": len(issues),
        "n_error_issues": int(sum(1 for x in issues if x["severity"] == "error")),
        "n_warning_issues": int(sum(1 for x in issues if x["severity"] == "warning")),
    }

    return {
        "summary": summary,
        "issues": issues,
        "inventory_rows": build_rule_inventory_rows(rules),
    }
