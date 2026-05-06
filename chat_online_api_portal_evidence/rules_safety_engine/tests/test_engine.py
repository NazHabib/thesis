import pandas as pd

from rules_safety_engine.rules_engine.engine import evaluate_rules
from rules_safety_engine.rules_engine.models import Rule
from rules_safety_engine.rules_engine.rulebase_audit import audit_rulebase



def test_engine_basic_htn_crisis():
    df = pd.DataFrame({
        "ParticipantKey": ["P1", "P2", "P3"],
        "Cycle": ["2015-2016", "2015-2016", "2015-2016"],
        "SBP_mean": [129, 180, 142],
        "DBP_mean": [79, 100, 92],
        "LBXGH": [5.5, 6.2, 10.5],
        "DR1TSODI": [2000, 2500, 3000],
        "Sodium_over_2300_flag": [0, 1, 1],
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
        ),
        Rule(
            rule_id="SODIUM_REDUCE",
            rule_version="1.0.0",
            name="Reduce sodium",
            group="lifestyle",
            priority=50,
            condition="DR1TSODI > 2300",
            missing_policy="na_false",
            severity="moderate",
            action_type="recommend",
            action_code="REDUCE_SODIUM",
            tags=["diet", "htn"],
            explanation_template="Reduce sodium intake.",
        ),
    ]

    result = evaluate_rules(df, rules)

    assert len(result.fired_rules) == 3
    assert any(r.action_code == "ER_REFERRAL" for r in result.recommendations)
    assert any(r.action_code == "REDUCE_SODIUM" for r in result.recommendations)

    p2_recs = [r for r in result.recommendations if r.participant_key == "P2"]
    assert any(r.action_code == "ER_REFERRAL" for r in p2_recs)
    assert not any(r.action_code == "REDUCE_SODIUM" for r in p2_recs), "critical escalation should suppress recommend"


def test_duplicate_action_code_keeps_strongest():
    df = pd.DataFrame({
        "ParticipantKey": ["P1"],
        "Cycle": ["2015-2016"],
        "SBP_mean": [150],
        "DBP_mean": [95],
    })

    rules = [
        Rule(
            rule_id="HTN_STAGE2_A",
            rule_version="1.0.0",
            name="Stage 2 A",
            group="htn",
            priority=100,
            condition="SBP_mean >= 140 or DBP_mean >= 90",
            missing_policy="na_false",
            severity="high",
            action_type="recommend",
            action_code="FOLLOWUP_HTN",
            tags=["htn"],
            explanation_template="Follow-up for hypertension.",
        ),
        Rule(
            rule_id="HTN_STAGE2_B",
            rule_version="1.0.0",
            name="Stage 2 B",
            group="htn",
            priority=80,
            condition="SBP_mean >= 140 or DBP_mean >= 90",
            missing_policy="na_false",
            severity="moderate",
            action_type="recommend",
            action_code="FOLLOWUP_HTN",
            tags=["htn"],
            explanation_template="Follow-up for hypertension.",
        ),
    ]

    result = evaluate_rules(df, rules)
    recs = [r for r in result.recommendations if r.participant_key == "P1"]
    assert len(recs) == 1
    assert recs[0].source_rule_ids == ["HTN_STAGE2_A"]
    
def test_conflict_group_keep_strongest():
    df = pd.DataFrame({
        "ParticipantKey": ["P1"],
        "Cycle": ["2015-2016"],
        "SBP_mean": [145],
        "DBP_mean": [95],
    })

    rules = [
        Rule(
            rule_id="R1",
            rule_version="1.0.0",
            name="Moderate sodium advice",
            group="diet",
            priority=50,
            condition="SBP_mean >= 140",
            missing_policy="na_false",
            severity="moderate",
            action_type="recommend",
            action_code="REDUCE_SODIUM_MODERATE",
            conflict_group="sodium_management",
            conflict_policy="keep_strongest",
        ),
        Rule(
            rule_id="R2",
            rule_version="1.0.0",
            name="Strong sodium advice",
            group="diet",
            priority=100,
            condition="SBP_mean >= 140",
            missing_policy="na_false",
            severity="high",
            action_type="recommend",
            action_code="REDUCE_SODIUM_STRONG",
            conflict_group="sodium_management",
            conflict_policy="keep_strongest",
        ),
    ]

    result = evaluate_rules(df, rules)
    codes = [r.action_code for r in result.recommendations]

    assert "REDUCE_SODIUM_STRONG" in codes
    assert "REDUCE_SODIUM_MODERATE" not in codes
    assert len(result.conflicts) >= 1


def test_explicit_suppression():
    df = pd.DataFrame({
        "ParticipantKey": ["P1"],
        "Cycle": ["2015-2016"],
        "SBP_mean": [185],
        "DBP_mean": [121],
        "HTN_category": ["Stage2"],
    })

    rules = [
        Rule(
            rule_id="CRISIS",
            rule_version="1.0.0",
            name="Crisis",
            group="safety",
            priority=1000,
            condition="SBP_mean >= 180 or DBP_mean >= 120",
            missing_policy="na_false",
            severity="critical",
            action_type="escalate",
            action_code="ER_REFERRAL",
            suppresses=["FOLLOWUP_HTN_STAGE2"],
        ),
        Rule(
            rule_id="FOLLOWUP",
            rule_version="1.0.0",
            name="Followup",
            group="htn",
            priority=100,
            condition='HTN_category == "Stage2"',
            missing_policy="na_false",
            severity="high",
            action_type="recommend",
            action_code="FOLLOWUP_HTN_STAGE2",
        ),
    ]

    result = evaluate_rules(df, rules)
    codes = [r.action_code for r in result.recommendations]

    assert "ER_REFERRAL" in codes
    assert "FOLLOWUP_HTN_STAGE2" not in codes
    assert len(result.conflicts) >= 1

def test_forbid_suppresses_same_tag_recommend():
    df = pd.DataFrame({
        "ParticipantKey": ["P1"],
        "Cycle": ["2015-2016"],
        "SBP_mean": [150],
        "DBP_mean": [95],
        "BMXBMI": [42],
    })

    rules = [
        Rule(
            rule_id="AVOID_INTENSE_EXERCISE",
            rule_version="1.0.0",
            name="Avoid intense exercise now",
            group="safety",
            priority=900,
            condition="SBP_mean >= 140",
            missing_policy="na_false",
            severity="high",
            action_type="forbid",
            action_code="AVOID_INTENSE_EXERCISE",
            tags=["exercise", "htn"],
            explanation_template="Avoid intense exercise until blood pressure is assessed.",
        ),
        Rule(
            rule_id="EXERCISE_MORE",
            rule_version="1.0.0",
            name="Increase exercise",
            group="lifestyle",
            priority=50,
            condition="BMXBMI >= 30",
            missing_policy="na_false",
            severity="moderate",
            action_type="recommend",
            action_code="INCREASE_ACTIVITY",
            tags=["exercise", "obesity"],
            explanation_template="Increase physical activity.",
        ),
    ]

    result = evaluate_rules(df, rules)
    recs = [r.action_code for r in result.recommendations if r.participant_key == "P1"]

    assert "AVOID_INTENSE_EXERCISE" in recs
    assert "INCREASE_ACTIVITY" not in recs
    

def test_rulebase_audit_detects_duplicate_rule_id():
    rules = [
        Rule(
            rule_id="R1",
            rule_version="1.0.0",
            name="Rule 1",
            group="test",
            priority=10,
            condition="SBP_mean >= 140",
            action_type="recommend",
            action_code="A1",
        ),
        Rule(
            rule_id="R1",
            rule_version="1.0.1",
            name="Rule 1 updated",
            group="test",
            priority=20,
            condition="SBP_mean >= 150",
            action_type="recommend",
            action_code="A2",
        ),
    ]

    audit = audit_rulebase(rules)
    assert any(x["issue_type"] == "duplicate_rule_id" for x in audit["issues"])


def test_rulebase_audit_detects_missing_explanation_template():
    rules = [
        Rule(
            rule_id="R1",
            rule_version="1.0.0",
            name="Rule 1",
            group="test",
            priority=10,
            condition="SBP_mean >= 140",
            action_type="recommend",
            action_code="A1",
            explanation_template="",
        )
    ]

    audit = audit_rulebase(rules)
    assert any(x["issue_type"] == "missing_explanation_template" for x in audit["issues"])
    
    
def test_critical_escalation_ranks_above_high_forbid():
    df = pd.DataFrame({
        "ParticipantKey": ["P1"],
        "SBP_mean": [185],
        "DBP_mean": [121],
        "BMXBMI": [42],
    })

    rules = [
        Rule(
            rule_id="CRITICAL_ESCALATE",
            rule_version="1.0.0",
            name="Critical escalation",
            group="safety",
            priority=100,
            condition="SBP_mean >= 180 or DBP_mean >= 120",
            missing_policy="na_false",
            severity="critical",
            action_type="escalate",
            action_code="URGENT_REVIEW",
            conflict_group="SAFETY_TEST",
            conflict_policy="keep_strongest",
        ),
        Rule(
            rule_id="HIGH_FORBID",
            rule_version="1.0.0",
            name="High forbid",
            group="safety",
            priority=999,
            condition="BMXBMI >= 40",
            missing_policy="na_false",
            severity="high",
            action_type="forbid",
            action_code="AVOID_UNSUPERVISED_INTENSE_ACTIVITY",
            conflict_group="SAFETY_TEST",
            conflict_policy="keep_strongest",
        ),
    ]

    result = evaluate_rules(df, rules)
    codes = [r.action_code for r in result.recommendations]

    assert "URGENT_REVIEW" in codes
    assert "AVOID_UNSUPERVISED_INTENSE_ACTIVITY" not in codes
