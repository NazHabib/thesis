import pandas as pd
from rules_safety_engine.rules_engine.models import Rule
from rules_safety_engine.rules_engine.engine import evaluate_rules

from rules_safety_engine.rules_engine.parser import (
    RuleParseError,
    apply_missing_policy,
    compile_condition,
    evaluate_condition,
)


def test_basic_compare():
    df = pd.DataFrame({"SBP_mean": [129, 130, 180]})
    mask = evaluate_condition("SBP_mean >= 130", df)
    assert mask.tolist() == [False, True, True]


def test_and_or():
    df = pd.DataFrame({
        "SBP_mean": [129, 130, 180],
        "DBP_mean": [79, 79, 100],
    })
    mask = evaluate_condition("SBP_mean >= 180 or DBP_mean >= 100", df)
    assert mask.tolist() == [False, False, True]


def test_in_operator():
    df = pd.DataFrame({"HTN_category": ["Normal", "Stage1", "Stage2"]})
    mask = evaluate_condition('HTN_category in ["Stage1", "Stage2"]', df)
    assert mask.tolist() == [False, True, True]


def test_between():
    df = pd.DataFrame({"BMXBMI": [22, 27, 33]})
    mask = evaluate_condition("between(BMXBMI, 25, 30)", df)
    assert mask.tolist() == [False, True, False]


def test_isna():
    df = pd.DataFrame({"DR1TSODI": [2301, None, 1000]})
    mask = evaluate_condition("isna(DR1TSODI)", df)
    assert mask.tolist() == [False, True, False]


def test_missing_policy_false():
    mask = pd.Series([True, pd.NA, False], dtype="boolean")
    out = apply_missing_policy(mask, "na_false")
    assert out.tolist() == [True, False, False]


def test_missing_policy_true():
    mask = pd.Series([True, pd.NA, False], dtype="boolean")
    out = apply_missing_policy(mask, "na_true")
    assert out.tolist() == [True, True, False]


def test_missing_policy_error():
    mask = pd.Series([True, pd.NA, False], dtype="boolean")
    try:
        apply_missing_policy(mask, "na_error", "X > 1")
        assert False, "Expected RuleParseError"
    except RuleParseError:
        assert True


def test_reject_disallowed_call():
    try:
        compile_condition('__import__("os").system("echo hacked")')
        assert False, "Expected RuleParseError"
    except RuleParseError:
        assert True
        
def test_atomic_condition_extraction():
    compiled = compile_condition("SBP_mean >= 180 or DBP_mean >= 120")
    atomics = compiled.atomic_conditions

    assert len(atomics) == 2
    assert atomics[0].feature == "SBP_mean"
    assert atomics[0].operator == ">="
    assert atomics[0].threshold == 180
    
def test_engine_evidence_contains_operator_and_threshold():
    df = pd.DataFrame({
        "ParticipantKey": ["P1"],
        "Cycle": ["2015-2016"],
        "SBP_mean": [185],
        "DBP_mean": [100],
    })

    rules = [
        Rule(
            rule_id="HTN_CRISIS_ESCALATE",
            rule_version="1.0.0",
            name="Escalate hypertensive crisis",
            group="safety",
            priority=1000,
            condition="SBP_mean >= 180 or DBP_mean >= 120",
            missing_policy="na_false",
            severity="critical",
            action_type="escalate",
            action_code="ER_REFERRAL",
            tags=["htn", "safety"],
            explanation_template="Blood pressure crisis threshold met.",
        )
    ]

    result = evaluate_rules(df, rules)
    assert len(result.fired_rules) == 1

    ev = result.fired_rules[0].evidence
    assert any(e.feature == "SBP_mean" and e.operator == ">=" and e.threshold == 180 for e in ev)
