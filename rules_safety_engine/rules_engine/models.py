from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


SeverityLevel = Literal["low", "moderate", "high", "critical"]
ActionType = Literal["recommend", "forbid", "escalate", "inform"]
MissingPolicy = Literal["na_false", "na_true", "na_error"]


@dataclass(frozen=True)
class Rule:
    """
    Canonical representation of one deterministic guideline/safety rule.
    """

    rule_id: str
    rule_version: str
    name: str
    group: str
    priority: int

    condition: str
    missing_policy: MissingPolicy = "na_false"

    severity: SeverityLevel = "moderate"
    action_type: ActionType = "recommend"
    action_code: str = ""

    tags: List[str] = field(default_factory=list)
    contraindications_unknown: bool = False

    explanation_template: str = ""
    citations: List[str] = field(default_factory=list)

    # New conflict/arbitration metadata
    conflict_group: Optional[str] = None
    conflict_policy: Optional[str] = None
    suppresses: List[str] = field(default_factory=list)

    provenance: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "unknown"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    enabled: bool = True

    def validate(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("rule_id cannot be empty")
        if not self.rule_version.strip():
            raise ValueError(f"{self.rule_id}: rule_version cannot be empty")
        if not self.name.strip():
            raise ValueError(f"{self.rule_id}: name cannot be empty")
        if not self.condition.strip():
            raise ValueError(f"{self.rule_id}: condition cannot be empty")
        if self.priority < 0:
            raise ValueError(f"{self.rule_id}: priority must be >= 0")
        if self.action_type not in {"recommend", "forbid", "escalate", "inform"}:
            raise ValueError(f"{self.rule_id}: invalid action_type={self.action_type}")
        if self.missing_policy not in {"na_false", "na_true", "na_error"}:
            raise ValueError(f"{self.rule_id}: invalid missing_policy={self.missing_policy}")
        if self.severity not in {"low", "moderate", "high", "critical"}:
            raise ValueError(f"{self.rule_id}: invalid severity={self.severity}")

        allowed_conflict_policies = {
            "keep_strongest",
            "keep_all",
            "suppress_weaker_same_group",
            "suppress_recommend_if_escalate",
            "mutually_exclusive",
            None,
        }
        if self.conflict_policy not in allowed_conflict_policies:
            raise ValueError(
                f"{self.rule_id}: invalid conflict_policy={self.conflict_policy}"
            )

        if not isinstance(self.tags, list):
            raise ValueError(f"{self.rule_id}: tags must be a list")
        if not isinstance(self.citations, list):
            raise ValueError(f"{self.rule_id}: citations must be a list")
        if not isinstance(self.suppresses, list):
            raise ValueError(f"{self.rule_id}: suppresses must be a list")


@dataclass
class EvidenceItem:
    """
    One structured piece of evidence used to explain why a rule fired.
    """

    feature: str
    value: Any
    operator: Optional[str] = None
    threshold: Optional[Any] = None
    passed: Optional[bool] = None


@dataclass
class FiredRule:
    """
    One rule firing for one participant/record.
    """

    participant_key: str
    rule_id: str
    rule_version: str
    action_type: ActionType
    action_code: str
    severity: SeverityLevel
    priority: int
    group: str

    fired: bool
    explanation: str = ""
    tags: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)

    cycle: Optional[str] = None
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Recommendation:
    """
    Final post-arbitration recommendation shown to downstream systems/UI.
    """

    participant_key: str
    action_code: str
    action_type: ActionType
    severity: SeverityLevel
    priority: int

    title: str = ""
    explanation: str = ""
    source_rule_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)

    suppressed_by: List[str] = field(default_factory=list)
    cycle: Optional[str] = None
    ruleset_version: Optional[str] = None
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ConflictRecord:
    """
    Captures conflict detection and arbitration decisions.
    """

    participant_key: str
    kept_action_code: str
    dropped_action_codes: List[str] = field(default_factory=list)
    reason: str = ""
    source_rule_ids: List[str] = field(default_factory=list)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class EngineManifest:
    """
    Run-level metadata for reproducibility and audit.
    """

    engine_version: str
    ruleset_version: str
    input_path: Optional[str] = None
    rows_processed: int = 0
    rules_loaded: int = 0
    rules_enabled: int = 0
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineRunResult:
    """
    Top-level output object returned by the engine.
    """

    manifest: EngineManifest
    fired_rules: List[FiredRule] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    conflicts: List[ConflictRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
