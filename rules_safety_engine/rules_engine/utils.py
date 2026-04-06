from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from jsonschema import Draft202012Validator

import numpy as np

import pandas as pd
import yaml

from .models import EngineManifest, Rule


# =========================
# Logging
# =========================

def setup_logger(name: str = "rules_safety_engine", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# =========================
# Path helpers
# =========================

def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =========================
# Serialization helpers
# =========================

def to_plain_dict(obj: Any) -> Any:
    """
    Convert dataclasses / pandas / numpy objects into JSON-safe plain Python objects.
    """

    if is_dataclass(obj):
        return {k: to_plain_dict(v) for k, v in asdict(obj).items()}

    if isinstance(obj, dict):
        return {str(k): to_plain_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_plain_dict(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    # pandas Series / Index
    if isinstance(obj, pd.Series):
        return [to_plain_dict(v) for v in obj.tolist()]
    if isinstance(obj, pd.Index):
        return [to_plain_dict(v) for v in obj.tolist()]

    # numpy scalar types -> native Python
    if isinstance(obj, np.generic):
        return obj.item()

    # pandas timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # native missing handling
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_plain_dict(obj), f, indent=indent, ensure_ascii=False)
    return path


def write_jsonl(records: Iterable[Any], path: str | Path) -> Path:
    path = ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(to_plain_dict(rec), ensure_ascii=False) + "\n")
    return path


def write_csv(records: Sequence[Any], path: str | Path) -> Path:
    path = ensure_parent_dir(path)
    rows = [to_plain_dict(r) for r in records]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# =========================
# Ruleset loading
# =========================

def load_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ruleset_file(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        payload = load_yaml(path)
    elif suffix == ".json":
        payload = load_json(path)
    else:
        raise ValueError(f"Unsupported ruleset format: {path}")

    if payload is None:
        return []

    # normalize list payload into {"rules": [...]}
    if isinstance(payload, list):
        payload = {"rules": payload}

    if not isinstance(payload, dict) or "rules" not in payload:
        raise ValueError("Ruleset file must be either a list of rules or a dict with key 'rules'")

    schema_path = Path(__file__).resolve().parent.parent / "rules" / "rule_schema.json"
    validate_ruleset_schema(payload, schema_path)

    rules = payload["rules"]
    if not isinstance(rules, list):
        raise ValueError("'rules' must be a list")

    return rules


# =========================
# Rule validation / parsing
# =========================

REQUIRED_RULE_FIELDS = {
    "rule_id",
    "rule_version",
    "name",
    "group",
    "priority",
    "condition",
    "missing_policy",
    "severity",
    "action_type",
    "action_code",
}


def validate_rule_dict(rule_dict: Dict[str, Any]) -> None:
    missing = REQUIRED_RULE_FIELDS - set(rule_dict.keys())
    if missing:
        raise ValueError(
            f"Rule missing required fields: {sorted(missing)} | rule_id={rule_dict.get('rule_id', '<unknown>')}"
        )

    if not isinstance(rule_dict["priority"], int):
        raise ValueError(f"{rule_dict.get('rule_id')}: priority must be int")

    if "tags" in rule_dict and rule_dict["tags"] is not None and not isinstance(rule_dict["tags"], list):
        raise ValueError(f"{rule_dict.get('rule_id')}: tags must be a list")

    if "citations" in rule_dict and rule_dict["citations"] is not None and not isinstance(rule_dict["citations"], list):
        raise ValueError(f"{rule_dict.get('rule_id')}: citations must be a list")

    if "suppresses" in rule_dict and rule_dict["suppresses"] is not None and not isinstance(rule_dict["suppresses"], list):
        raise ValueError(f"{rule_dict.get('rule_id')}: suppresses must be a list")

    if "provenance" in rule_dict and rule_dict["provenance"] is not None and not isinstance(rule_dict["provenance"], dict):
        raise ValueError(f"{rule_dict.get('rule_id')}: provenance must be a dict")

    if "conflict_group" in rule_dict and rule_dict["conflict_group"] is not None and not isinstance(rule_dict["conflict_group"], str):
        raise ValueError(f"{rule_dict.get('rule_id')}: conflict_group must be a string or null")

    allowed_conflict_policies = {
        "keep_strongest",
        "keep_all",
        "suppress_weaker_same_group",
        "suppress_recommend_if_escalate",
        "mutually_exclusive",
        None,
    }
    if rule_dict.get("conflict_policy") not in allowed_conflict_policies:
        raise ValueError(
            f"{rule_dict.get('rule_id')}: invalid conflict_policy={rule_dict.get('conflict_policy')}"
        )


def rule_from_dict(rule_dict: Dict[str, Any]) -> Rule:
    validate_rule_dict(rule_dict)

    rule = Rule(
        rule_id=rule_dict["rule_id"],
        rule_version=rule_dict["rule_version"],
        name=rule_dict["name"],
        group=rule_dict["group"],
        priority=rule_dict["priority"],
        condition=rule_dict["condition"],
        missing_policy=rule_dict.get("missing_policy", "na_false"),
        severity=rule_dict.get("severity", "moderate"),
        action_type=rule_dict.get("action_type", "recommend"),
        action_code=rule_dict.get("action_code", ""),
        tags=rule_dict.get("tags", []) or [],
        contraindications_unknown=bool(rule_dict.get("contraindications_unknown", False)),
        explanation_template=rule_dict.get("explanation_template", ""),
        citations=rule_dict.get("citations", []) or [],
        conflict_group=rule_dict.get("conflict_group"),
        conflict_policy=rule_dict.get("conflict_policy"),
        suppresses=rule_dict.get("suppresses", []) or [],
        provenance=rule_dict.get("provenance", {}) or {},
        created_by=rule_dict.get("created_by", "unknown"),
        created_at=rule_dict.get("created_at") or Rule.__dataclass_fields__["created_at"].default_factory(),
        enabled=bool(rule_dict.get("enabled", True)),
    )
    rule.validate()
    return rule


def load_rules(path: str | Path) -> List[Rule]:
    raw_rules = load_ruleset_file(path)
    return [rule_from_dict(r) for r in raw_rules]


def sort_rules(rules: Sequence[Rule]) -> List[Rule]:
    return sorted(rules, key=lambda r: (-r.priority, r.rule_id))


def filter_enabled_rules(rules: Sequence[Rule]) -> List[Rule]:
    return [r for r in rules if r.enabled]


# =========================
# Manifest helpers
# =========================

def manifest_to_dict(manifest: EngineManifest) -> Dict[str, Any]:
    return to_plain_dict(manifest)


def write_manifest(manifest: EngineManifest, path: str | Path) -> Path:
    return write_json(manifest_to_dict(manifest), path)


# =========================
# DataFrame helpers
# =========================

def require_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")


def coerce_numeric_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
    
def load_rule_schema(schema_path: str | Path) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_ruleset_schema(payload: Dict[str, Any], schema_path: str | Path) -> None:
    schema = load_rule_schema(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))

    if errors:
        messages = []
        for err in errors:
            path = ".".join(str(x) for x in err.path) if err.path else "<root>"
            messages.append(f"{path}: {err.message}")
        raise ValueError("Ruleset schema validation failed:\n" + "\n".join(messages))


def row_count(df: pd.DataFrame) -> int:
    return int(df.shape[0])


def preview_columns(df: pd.DataFrame, n: int = 20) -> List[str]:
    return list(df.columns[:n])
