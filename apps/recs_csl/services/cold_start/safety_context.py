import numpy as np
import pandas as pd

from .config import CONF_HIGH, CONF_LOW, CONF_MEDIUM, COLD_START_VERSION


def _build_warning_lists(out: pd.DataFrame) -> list[list[str]]:
    warnings_per_row = []

    for _, row in out.iterrows():
        codes = []

        if row["triage_reason"] == "hypertensive_crisis":
            codes.append("CS_TRIAGE_HTN_CRISIS")
        elif row["triage_reason"] == "severe_hyperglycemia":
            codes.append("CS_TRIAGE_SEVERE_HYPERGLYCEMIA")
        elif row["triage_reason"] == "both":
            codes.append("CS_TRIAGE_HTN_CRISIS")
            codes.append("CS_TRIAGE_SEVERE_HYPERGLYCEMIA")

        if row["bp_quality"] == "low":
            codes.append("CS_WARN_BP_LOW_QUALITY")
        elif row["bp_quality"] == "unknown":
            codes.append("CS_WARN_BP_MISSING")

        if row["diet_day1_status"] in ["not_reliable", "unknown"]:
            codes.append("CS_WARN_DIET_RECALL_LIMITED")

        if not row["a1c_core"]:
            codes.append("CS_WARN_MISSING_A1C")

        if not row["bmi_core"]:
            codes.append("CS_WARN_MISSING_BMI")

        if not row["bp_core"]:
            codes.append("CS_WARN_MISSING_BP")

        warnings_per_row.append(codes)

    return warnings_per_row


def _build_note_lists(out: pd.DataFrame) -> list[list[str]]:
    notes_per_row = []

    for _, row in out.iterrows():
        notes = []

        if row["feature_tier"] == "minimal":
            notes.append(
                "Perfil em cold-start: existem dados clínicos essenciais em falta; a confiança global foi limitada."
            )

        if row["diet_day1_status"] in ["not_reliable", "unknown"]:
            notes.append(
                "Os dados dietéticos do Dia 1 são limitados ou não fiáveis; recomendações dietéticas específicas devem ser interpretadas com cautela."
            )

        if row["bp_quality"] == "low":
            notes.append(
                "A qualidade da medição da pressão arterial é limitada; idealmente confirmar com mais medições."
            )

        if row["triage_mode"]:
            notes.append(
                "Modo de triagem ativo: recomendações lifestyle detalhadas foram suprimidas em favor de orientação de segurança."
            )

        notes_per_row.append(notes)

    return notes_per_row


def build_safety_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Hypertensive_crisis_flag"] = out["Hypertensive_crisis_flag"].fillna(0).astype("int8")
    out["Severe_hyperglycemia_flag"] = out["Severe_hyperglycemia_flag"].fillna(0).astype("int8")

    out["triage_mode"] = (
        (out["Hypertensive_crisis_flag"] == 1)
        | (out["Severe_hyperglycemia_flag"] == 1)
    )

    out["triage_reason"] = np.select(
        [
            (~out["triage_mode"]),
            (out["Hypertensive_crisis_flag"] == 1) & (out["Severe_hyperglycemia_flag"] == 0),
            (out["Hypertensive_crisis_flag"] == 0) & (out["Severe_hyperglycemia_flag"] == 1),
            (out["Hypertensive_crisis_flag"] == 1) & (out["Severe_hyperglycemia_flag"] == 1),
        ],
        ["none", "hypertensive_crisis", "severe_hyperglycemia", "both"],
        default="none",
    )

    bp_present = out["SBP_mean"].notna() & out["DBP_mean"].notna()
    bp_n_ok = out["BP_n_used"].fillna(0).astype("int16") >= 2

    out["bp_quality"] = np.select(
        [bp_present & bp_n_ok, bp_present & ~bp_n_ok, ~bp_present],
        ["good", "low", "unknown"],
        default="unknown",
    )

    out["diet_day1_status"] = np.select(
        [
            out["DietDay1_reliable_flag"].isna(),
            out["DietDay1_reliable_flag"].fillna(0).astype("int8").eq(1),
            out["DietDay1_reliable_flag"].fillna(0).astype("int8").eq(0),
        ],
        ["unknown", "reliable", "not_reliable"],
        default="unknown",
    )

    out["confidence_ceiling_global"] = np.select(
        [out["feature_tier"].eq("minimal")],
        [CONF_MEDIUM],
        default=CONF_HIGH,
    )

    out["ceiling_DietPattern"] = np.where(out["diet_day1_status"].eq("reliable"), CONF_HIGH, CONF_LOW)
    out["ceiling_Movement"] = np.where(out["Any_MVPA_flag"].notna(), CONF_MEDIUM, CONF_LOW)
    out["ceiling_Weight"] = np.where(out["bmi_core"], CONF_HIGH, CONF_LOW)
    out["ceiling_Smoking"] = np.where(out["Current_smoker_flag"].notna(), CONF_HIGH, CONF_LOW)
    out["ceiling_Alcohol"] = CONF_MEDIUM
    out["ceiling_Lipids"] = np.where(out["feature_tier"].isin(["core", "full"]), CONF_MEDIUM, CONF_LOW)
    out["ceiling_Monitoring"] = np.where(out["bp_quality"].isin(["low", "unknown"]), CONF_MEDIUM, CONF_HIGH)
    out["ceiling_IntegratedCardiometabolic"] = np.select(
        [out["feature_tier"].eq("full"), out["feature_tier"].eq("core")],
        [CONF_HIGH, CONF_MEDIUM],
        default=CONF_LOW,
    )

    out["warnings"] = _build_warning_lists(out)
    out["notes_pt"] = _build_note_lists(out)

    out["cold_start_version"] = COLD_START_VERSION
    return out
