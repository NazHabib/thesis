from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .models import (
    ConflictRecord,
    EngineManifest,
    EngineRunResult,
    EvidenceItem,
    FiredRule,
    Recommendation,
    Rule,
)
from .parser import CompiledExpression, RuleParseError, apply_missing_policy, compile_condition
from .utils import (
    filter_enabled_rules,
    require_columns,
    row_count,
    sort_rules,
)


ACTION_RANK = {
    "recommend": 1,
    "inform": 1,
    "forbid": 3,
    "escalate": 4,
}

SEVERITY_RANK = {
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}


def _action_arbitration_rank(action_type: str, severity: str, priority: int) -> Tuple[int, int, int]:
    """
    Higher tuple wins.
    Safety-first:
      critical escalate > forbid > escalate > recommend/inform
    """
    action_base = ACTION_RANK.get(action_type, 0)
    severity_base = SEVERITY_RANK.get(severity, 0)
    return (action_base, severity_base, priority)


def _build_title(action_code: str) -> str:
    return action_code.replace("_", " ").strip().title()


def _safe_render_template(template: str, row: pd.Series) -> str:
    """
    Very conservative string formatting using only row keys.
    Unknown keys are left untouched.
    """
    if not template:
        return ""

    data = {str(k): row.get(k) for k in row.index}

    class _SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    try:
        return template.format_map(_SafeDict(data))
    except Exception:
        return template


def _extract_feature_names(compiled: CompiledExpression) -> List[str]:
    return list(compiled.feature_names)


def _make_evidence_items(row: pd.Series, compiled: CompiledExpression) -> List[EvidenceItem]:
    evidence: List[EvidenceItem] = []

    used_features = set()

    for atomic in compiled.atomic_conditions:
        if atomic.feature in row.index:
            value = row[atomic.feature]
            passed = _evaluate_atomic_pass(value, atomic.operator, atomic.threshold)
            evidence.append(
                EvidenceItem(
                    feature=atomic.feature,
                    value=value,
                    operator=atomic.operator,
                    threshold=atomic.threshold,
                    passed=passed,
                )
            )
            used_features.add(atomic.feature)

    # fallback: include remaining referenced features even if not in atomic_conditions
    for feat in compiled.feature_names:
        if feat in row.index and feat not in used_features:
            evidence.append(
                EvidenceItem(
                    feature=feat,
                    value=row[feat],
                )
            )

    return evidence
    
def _evaluate_atomic_pass(value: Any, operator: str, threshold: Any) -> Optional[bool]:
    try:
        if pd.isna(value):
            return None

        if operator == "==":
            return value == threshold
        if operator == "!=":
            return value != threshold
        if operator == "<":
            return value < threshold
        if operator == "<=":
            return value <= threshold
        if operator == ">":
            return value > threshold
        if operator == ">=":
            return value >= threshold
        if operator == "in":
            return value in threshold if threshold is not None else None
        if operator == "not in":
            return value not in threshold if threshold is not None else None
    except Exception:
        return None

    return None


def _ensure_participant_key(df: pd.DataFrame) -> None:
    require_columns(df, ["ParticipantKey"])


def _ensure_cycle_optional(row: pd.Series) -> Optional[str]:
    return str(row["Cycle"]) if "Cycle" in row.index and pd.notna(row["Cycle"]) else None
    
def _recommendation_from_fired_rule(r: FiredRule, ruleset_version: str) -> Recommendation:
    rec = Recommendation(
        participant_key=r.participant_key,
        action_code=r.action_code,
        action_type=r.action_type,
        severity=r.severity,
        priority=r.priority,
        title=_build_title(r.action_code),
        explanation=r.explanation,
        source_rule_ids=[r.rule_id],
        tags=list(r.tags),
        evidence=list(r.evidence),
        cycle=r.cycle,
        ruleset_version=ruleset_version,
    )
    return rec

def _strongest_rule(rules: Sequence[FiredRule]) -> FiredRule:
    return sorted(
        rules,
        key=lambda x: _action_arbitration_rank(x.action_type, x.severity, x.priority),
        reverse=True,
    )[0]


def _apply_explicit_suppressions(
    participant_key: str,
    candidate_rules: List[FiredRule],
) -> Tuple[List[FiredRule], List[ConflictRecord]]:
    """
    Apply explicit suppressions declared in the originating Rule definition.
    Since FiredRule does not carry suppresses directly, engine will attach it
    dynamically when materializing FiredRule objects.
    """
    kept = list(candidate_rules)
    conflicts: List[ConflictRecord] = []

    by_action_code = {r.action_code: r for r in candidate_rules}
    dropped_codes = set()

    for r in candidate_rules:
        suppresses = getattr(r, "suppresses", []) or []
        for target_code in suppresses:
            if target_code in by_action_code and target_code != r.action_code:
                dropped_codes.add(target_code)
                conflicts.append(
                    ConflictRecord(
                        participant_key=participant_key,
                        kept_action_code=r.action_code,
                        dropped_action_codes=[target_code],
                        reason="explicit_rule_suppression",
                        source_rule_ids=[r.rule_id, by_action_code[target_code].rule_id],
                    )
                )

    kept = [r for r in kept if r.action_code not in dropped_codes]
    return kept, conflicts


def _apply_conflict_group_policies(
    participant_key: str,
    candidate_rules: List[FiredRule],
) -> Tuple[List[FiredRule], List[ConflictRecord]]:
    """
    Resolve conflicts inside declared conflict groups.

    Supported policies:
    - keep_all
    - keep_strongest
    - suppress_weaker_same_group
    - suppress_recommend_if_escalate
    - mutually_exclusive
    """
    grouped: Dict[str, List[FiredRule]] = defaultdict(list)
    passthrough: List[FiredRule] = []
    conflicts: List[ConflictRecord] = []

    for r in candidate_rules:
        group = getattr(r, "conflict_group", None)
        if group:
            grouped[group].append(r)
        else:
            passthrough.append(r)

    resolved: List[FiredRule] = list(passthrough)

    for group_name, rules in grouped.items():
        if len(rules) == 1:
            resolved.extend(rules)
            continue

        policy = getattr(rules[0], "conflict_policy", None)

        if policy in {None, "keep_all"}:
            resolved.extend(rules)
            continue

        if policy in {"keep_strongest", "suppress_weaker_same_group", "mutually_exclusive"}:
            kept = _strongest_rule(rules)
            dropped = [r for r in rules if r is not kept]
            resolved.append(kept)

            conflicts.append(
                ConflictRecord(
                    participant_key=participant_key,
                    kept_action_code=kept.action_code,
                    dropped_action_codes=[r.action_code for r in dropped],
                    reason=f"conflict_group:{group_name}:{policy}",
                    source_rule_ids=[r.rule_id for r in rules],
                )
            )
            continue

        if policy == "suppress_recommend_if_escalate":
            escalates = [r for r in rules if r.action_type == "escalate"]
            if escalates:
                kept = _strongest_rule(escalates)
                dropped = [r for r in rules if r is not kept]
                resolved.append(kept)

                conflicts.append(
                    ConflictRecord(
                        participant_key=participant_key,
                        kept_action_code=kept.action_code,
                        dropped_action_codes=[r.action_code for r in dropped],
                        reason=f"conflict_group:{group_name}:suppress_recommend_if_escalate",
                        source_rule_ids=[r.rule_id for r in rules],
                    )
                )
            else:
                resolved.extend(rules)
            continue

        # fallback
        resolved.extend(rules)

    return resolved, conflicts

class RuleEngine:
    """
    Deterministic, vectorized rules & safety engine.

    Core flow:
      1) validate dataframe
      2) compile rules once
      3) evaluate rule masks over dataframe
      4) materialize fired rules
      5) resolve conflicts
      6) arbitrate to final recommendations
      7) return audit-ready result
    """

    def __init__(
        self,
        rules: Sequence[Rule],
        engine_version: str = "0.1.0",
        ruleset_version: Optional[str] = None,
    ) -> None:
        self.engine_version = engine_version
        self.rules = sort_rules(filter_enabled_rules(rules))
        self.ruleset_version = ruleset_version or self._infer_ruleset_version(self.rules)
        self._compiled: Dict[str, CompiledExpression] = {}
        self._compile_rules()

    @staticmethod
    def _infer_ruleset_version(rules: Sequence[Rule]) -> str:
        versions = sorted({r.rule_version for r in rules})
        if not versions:
            return "unknown"
        if len(versions) == 1:
            return versions[0]
        return ",".join(versions)

    def _compile_rules(self) -> None:
        compiled: Dict[str, CompiledExpression] = {}
        for rule in self.rules:
            compiled[rule.rule_id] = compile_condition(rule.condition)
        self._compiled = compiled

    def validate_input(self, df: pd.DataFrame) -> None:
        _ensure_participant_key(df)

    def evaluate(
        self,
        df: pd.DataFrame,
        input_path: Optional[str] = None,
    ) -> EngineRunResult:
        self.validate_input(df)

        fired_rules, warnings = self._evaluate_rules(df)
        recommendations, conflicts = self._arbitrate(df, fired_rules)

        manifest = EngineManifest(
            engine_version=self.engine_version,
            ruleset_version=self.ruleset_version,
            input_path=input_path,
            rows_processed=row_count(df),
            rules_loaded=len(self.rules),
            rules_enabled=len(self.rules),
            extra={
                "fired_rules_count": len(fired_rules),
                "recommendations_count": len(recommendations),
                "conflicts_count": len(conflicts),
            },
        )

        return EngineRunResult(
            manifest=manifest,
            fired_rules=fired_rules,
            recommendations=recommendations,
            conflicts=conflicts,
            warnings=warnings,
        )

    def _evaluate_rules(self, df: pd.DataFrame) -> Tuple[List[FiredRule], List[str]]:
        fired_rules: List[FiredRule] = []
        warnings: List[str] = []

        for rule in self.rules:
            compiled = self._compiled[rule.rule_id]

            try:
                raw_mask = compiled.evaluate(df)
                final_mask = apply_missing_policy(raw_mask, rule.missing_policy, rule.condition)
            except RuleParseError as e:
                warnings.append(f"{rule.rule_id}: {str(e)}")
                continue
            except Exception as e:
                warnings.append(f"{rule.rule_id}: unexpected evaluation error: {str(e)}")
                continue

            if not isinstance(final_mask, pd.Series):
                warnings.append(f"{rule.rule_id}: evaluation did not return a pandas Series")
                continue

            matched_idx = df.index[final_mask]

            for idx in matched_idx:
                row = df.loc[idx]
                participant_key = str(row["ParticipantKey"])
                explanation = _safe_render_template(rule.explanation_template, row)
                evidence = _make_evidence_items(row, compiled)

                fired = FiredRule(
                    participant_key=participant_key,
                    rule_id=rule.rule_id,
                    rule_version=rule.rule_version,
                    action_type=rule.action_type,
                    action_code=rule.action_code,
                    severity=rule.severity,
                    priority=rule.priority,
                    group=rule.group,
                    fired=True,
                    explanation=explanation,
                    tags=list(rule.tags),
                    evidence=evidence,
                    cycle=_ensure_cycle_optional(row),
                )

                # attach richer conflict metadata dynamically
                setattr(fired, "conflict_group", getattr(rule, "conflict_group", None))
                setattr(fired, "conflict_policy", getattr(rule, "conflict_policy", None))
                setattr(fired, "suppresses", list(getattr(rule, "suppresses", []) or []))

                fired_rules.append(fired)

        return fired_rules, warnings

    def _arbitrate(
        self,
        df: pd.DataFrame,
        fired_rules: Sequence[FiredRule],
    ) -> Tuple[List[Recommendation], List[ConflictRecord]]:
        """
        Arbitration strategy:
        1) group by participant
        2) within participant, deduplicate same action_code
        3) apply explicit suppressions
        4) apply conflict-group policies
        5) if any critical escalate exists, suppress recommend/inform
        6) forbid suppresses same-tag recommend/inform
        7) build final recommendations
        """
        by_participant: Dict[str, List[FiredRule]] = defaultdict(list)
        for fr in fired_rules:
            by_participant[fr.participant_key].append(fr)

        recommendations: List[Recommendation] = []
        conflicts: List[ConflictRecord] = []

        for participant_key, items in by_participant.items():
            # step A: deduplicate same action_code by strongest rule
            best_by_action: Dict[str, FiredRule] = {}
            duplicates: Dict[str, List[FiredRule]] = defaultdict(list)

            for item in items:
                duplicates[item.action_code].append(item)
                current = best_by_action.get(item.action_code)
                if current is None:
                    best_by_action[item.action_code] = item
                    continue

                if _action_arbitration_rank(
                    item.action_type, item.severity, item.priority
                ) > _action_arbitration_rank(
                    current.action_type, current.severity, current.priority
                ):
                    best_by_action[item.action_code] = item

            for action_code, group in duplicates.items():
                if len(group) > 1:
                    kept = best_by_action[action_code]
                    dropped = [g.rule_id for g in group if g.rule_id != kept.rule_id]
                    conflicts.append(
                        ConflictRecord(
                            participant_key=participant_key,
                            kept_action_code=action_code,
                            dropped_action_codes=[action_code] if dropped else [],
                            reason="duplicate_action_code_keep_strongest_rule",
                            source_rule_ids=[g.rule_id for g in group],
                        )
                    )

            candidate_rules = list(best_by_action.values())

            # step B: explicit suppressions
            candidate_rules, explicit_conflicts = _apply_explicit_suppressions(
                participant_key, candidate_rules
            )
            conflicts.extend(explicit_conflicts)

            # step C: conflict-group policies
            candidate_rules, group_conflicts = _apply_conflict_group_policies(
                participant_key, candidate_rules
            )
            conflicts.extend(group_conflicts)

            # step D: critical escalation suppresses recommend/inform
            has_critical_escalate = any(
                (r.action_type == "escalate" and r.severity == "critical")
                for r in candidate_rules
            )

            if has_critical_escalate:
                kept_rules: List[FiredRule] = []
                dropped_rules: List[FiredRule] = []

                for r in candidate_rules:
                    if r.action_type in {"recommend", "inform"}:
                        dropped_rules.append(r)
                    else:
                        kept_rules.append(r)

                if dropped_rules:
                    conflicts.append(
                        ConflictRecord(
                            participant_key=participant_key,
                            kept_action_code="CRITICAL_ESCALATION_SET",
                            dropped_action_codes=[r.action_code for r in dropped_rules],
                            reason="critical_escalation_suppresses_recommendations",
                            source_rule_ids=[r.rule_id for r in candidate_rules],
                        )
                    )
                candidate_rules = kept_rules

            # step E: forbid suppresses same-tag recommend/inform
            forbid_rules = [r for r in candidate_rules if r.action_type == "forbid"]
            if forbid_rules:
                kept_rules = []
                dropped_rules = []

                forbid_tags = set()
                for fr in forbid_rules:
                    forbid_tags.update(fr.tags)

                for r in candidate_rules:
                    if r.action_type in {"recommend", "inform"} and set(r.tags).intersection(forbid_tags):
                        dropped_rules.append(r)
                    else:
                        kept_rules.append(r)

                if dropped_rules:
                    conflicts.append(
                        ConflictRecord(
                            participant_key=participant_key,
                            kept_action_code="FORBID_SET",
                            dropped_action_codes=[r.action_code for r in dropped_rules],
                            reason="forbid_suppresses_same_tag_recommendations",
                            source_rule_ids=[r.rule_id for r in candidate_rules],
                        )
                    )
                candidate_rules = kept_rules

            # step F: final recommendations
            for r in sorted(
                candidate_rules,
                key=lambda x: _action_arbitration_rank(
                    x.action_type, x.severity, x.priority
                ),
                reverse=True,
            ):
                rec = _recommendation_from_fired_rule(r, self.ruleset_version)
                recommendations.append(rec)

        return recommendations, conflicts

def evaluate_rules(
    df: pd.DataFrame,
    rules: Sequence[Rule],
    input_path: Optional[str] = None,
    engine_version: str = "0.1.0",
    ruleset_version: Optional[str] = None,
) -> EngineRunResult:
    engine = RuleEngine(
        rules=rules,
        engine_version=engine_version,
        ruleset_version=ruleset_version,
    )
    return engine.evaluate(df=df, input_path=input_path)


def explain_participant(
    run_result: EngineRunResult,
    participant_key: str,
) -> Dict[str, Any]:
    participant_fired = [fr for fr in run_result.fired_rules if fr.participant_key == participant_key]
    participant_recs = [rec for rec in run_result.recommendations if rec.participant_key == participant_key]
    participant_conflicts = [c for c in run_result.conflicts if c.participant_key == participant_key]

    return {
        "participant_key": participant_key,
        "fired_rules": [asdict(x) for x in participant_fired],
        "recommendations": [asdict(x) for x in participant_recs],
        "conflicts": [asdict(x) for x in participant_conflicts],
        "manifest": asdict(run_result.manifest),
        "warnings": list(run_result.warnings),
    }
