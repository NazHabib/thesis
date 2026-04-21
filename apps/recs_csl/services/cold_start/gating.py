import numpy as np
import pandas as pd

from .config import (
    ALLOW_ONE_SUPPORT_ITEM_IN_TRIAGE,
    CONF_TO_INT,
    GENERATED_ESCALATE_CODE,
    INT_TO_CONF,
    SEVERITY_RANK,
)



def pick_family_ceiling(df: pd.DataFrame) -> pd.Series:
    fam = df["bundle_family"].astype("string").fillna("IntegratedCardiometabolic")

    return pd.Series(
        np.select(
            [
                fam.eq("DietPattern"),
                fam.eq("Movement"),
                fam.eq("Weight"),
                fam.eq("Smoking"),
                fam.eq("Alcohol"),
                fam.eq("Lipids"),
                fam.eq("Monitoring"),
                fam.eq("IntegratedCardiometabolic"),
            ],
            [
                df["ceiling_DietPattern"],
                df["ceiling_Movement"],
                df["ceiling_Weight"],
                df["ceiling_Smoking"],
                df["ceiling_Alcohol"],
                df["ceiling_Lipids"],
                df["ceiling_Monitoring"],
                df["ceiling_IntegratedCardiometabolic"],
            ],
            default="medium",
        ),
        index=df.index,
    )


def apply_cold_start_gating(
    candidates_df: pd.DataFrame,
    safety_context_df: pd.DataFrame,
) -> pd.DataFrame:
    df = candidates_df.merge(
        safety_context_df,
        on="ParticipantKey",
        how="left",
        validate="m:1",
    )
    
    if candidates_df.empty:
    out = candidates_df.copy()
    for col, default in {
        "coldstart_gate": pd.Series(dtype="object"),
        "coldstart_drop_reason": pd.Series(dtype="object"),
        "generated_by_layer": pd.Series(dtype="object"),
        "coldstart_confidence_ceiling": pd.Series(dtype="object"),
        "coldstart_note_codes": pd.Series(dtype="object"),
        "cold_start_flag": pd.Series(dtype="bool"),
    }.items():
        if col not in out.columns:
            out[col] = default
    return out

    df["coldstart_gate"] = "keep"
    df["coldstart_drop_reason"] = None
    df["generated_by_layer"] = None

    is_triage = df["triage_mode"].fillna(False)
    is_escalate = df["action_type"].astype("string").eq("escalate")

    df.loc[is_triage & ~is_escalate, "coldstart_gate"] = "drop"
    df.loc[is_triage & ~is_escalate, "coldstart_drop_reason"] = "triage_drop_non_escalate"

    if ALLOW_ONE_SUPPORT_ITEM_IN_TRIAGE:
        df["sev_rank"] = (
            df["severity"]
            .astype("string")
            .map(SEVERITY_RANK)
            .fillna(0)
            .astype("int8")
        )
        df["priority_num"] = (
            pd.to_numeric(df["priority"], errors="coerce")
            .fillna(0)
            .astype("int16")
        )

        monitoring_support_mask = (
            is_triage
            & ~is_escalate
            & df["bundle_family"].astype("string").eq("Monitoring")
        )

        inform_support_mask = (
            is_triage
            & ~is_escalate
            & df["action_type"].astype("string").eq("inform")
        )

        monitoring_keep = (
            df.loc[monitoring_support_mask]
            .sort_values(
                ["ParticipantKey", "sev_rank", "priority_num", "action_code"],
                ascending=[True, False, False, True],
            )
            .groupby("ParticipantKey", sort=False)
            .head(1)[["ParticipantKey", "action_code"]]
            .assign(_keep_support=1)
        )

        kept_participants = set(monitoring_keep["ParticipantKey"].astype(str).tolist())

        inform_keep = (
            df.loc[inform_support_mask & ~df["ParticipantKey"].astype(str).isin(kept_participants)]
            .sort_values(
                ["ParticipantKey", "sev_rank", "priority_num", "action_code"],
                ascending=[True, False, False, True],
            )
            .groupby("ParticipantKey", sort=False)
            .head(1)[["ParticipantKey", "action_code"]]
            .assign(_keep_support=1)
        )

        support_keep = pd.concat([monitoring_keep, inform_keep], ignore_index=True, sort=False)

        df = df.merge(support_keep, on=["ParticipantKey", "action_code"], how="left")

        restore = df["_keep_support"].fillna(0).astype("int8").eq(1)
        df.loc[restore, "coldstart_gate"] = "keep"
        df.loc[restore, "coldstart_drop_reason"] = None

        df = df.drop(columns=["_keep_support", "sev_rank", "priority_num"], errors="ignore")

    family_conf = pick_family_ceiling(df)
    global_i = df["confidence_ceiling_global"].map(CONF_TO_INT).fillna(2).astype("int8")
    family_i = family_conf.map(CONF_TO_INT).fillna(2).astype("int8")

    df["coldstart_confidence_ceiling"] = pd.Series(
        np.minimum(global_i.to_numpy(), family_i.to_numpy()),
        index=df.index,
    ).map(INT_TO_CONF)

    df.loc[
        df["action_type"].astype("string").eq("escalate") & df["triage_mode"].fillna(False),
        "coldstart_confidence_ceiling",
    ] = "high"

    df["cold_start_flag"] = df["feature_tier"].eq("minimal")

    triage_pks = set(df.loc[df["triage_mode"], "ParticipantKey"].unique())
    existing_escalate_pks = set(
        df.loc[
            (df["triage_mode"])
            & (df["action_type"].astype("string") == "escalate")
            & (df["coldstart_gate"] == "keep"),
            "ParticipantKey",
        ].unique()
    )
    missing_escalates = triage_pks - existing_escalate_pks

    if missing_escalates:
        base = safety_context_df[
            safety_context_df["ParticipantKey"].isin(missing_escalates)
        ][["ParticipantKey"]].copy()

        generated = base.assign(
            action_code=GENERATED_ESCALATE_CODE,
            action_type="escalate",
            severity="critical",
            priority=100,
            tags=[["monitoring"] for _ in range(len(base))],
            evidence=[
                [{
                    "feature": "triage_mode",
                    "value": True,
                    "operator": "==",
                    "threshold": True,
                    "passed": True,
                }]
                for _ in range(len(base))
            ],
            source_rule_ids=[["CSL_GENERATED"] for _ in range(len(base))],
            bundle_family="Monitoring",
            coldstart_gate="keep",
            coldstart_drop_reason=None,
            generated_by_layer="cold_start_safety",
            coldstart_confidence_ceiling="high",
            cold_start_flag=True,
        )

        generated = generated.merge(
            safety_context_df,
            on="ParticipantKey",
            how="left",
            validate="m:1",
        )

        df = pd.concat([df, generated], ignore_index=True, sort=False)

    df["coldstart_note_codes"] = [[] for _ in range(len(df))]

    for idx, row in df.iterrows():
        codes = []

        if row["feature_tier"] == "minimal":
            codes.append("CS_NOTE_COLD_START_MINIMAL")

        if row["bundle_family"] == "DietPattern" and row["diet_day1_status"] in ["not_reliable", "unknown"]:
            codes.append("CS_NOTE_DIET_RECALL_LIMITED")

        if row["bundle_family"] == "Monitoring" and row["bp_quality"] in ["low", "unknown"]:
            codes.append("CS_NOTE_BP_CHECK_RECOMMENDED")

        if row["action_type"] == "escalate" and row["triage_mode"]:
            codes.append("CS_NOTE_TRIAGE_ESCALATION")

        if row["generated_by_layer"] == "cold_start_safety":
            codes.append("CS_NOTE_GENERATED_ESCALATE")

        df.at[idx, "coldstart_note_codes"] = codes

    return df[df["coldstart_gate"] == "keep"].copy()
