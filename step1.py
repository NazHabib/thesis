#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical Target Engineering — NHANES
Fontes (coloca-as como referências na tese; URLs apenas em code comment):
- ACC/AHA 2017 HTA categories + lifestyle: https://www.acc.org/Latest-in-Cardiology/ten-points-to-remember/2017/11/09/11/41/2017-Guideline-for-High-Blood-Pressure-in-Adults
- AHA Hypertensive crisis threshold: https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings/when-to-call-911-for-high-blood-pressure
- ADA A1c diagnosis: https://diabetes.org/about-diabetes/a1c
- ADA severe hyperglycemia note (A1c >10): https://diabetesjournals.org/care/article/49/Supplement_1/S183/163934/9-Pharmacologic-Approaches-to-Glycemic-Treatment
- CDC Adult BMI classes: https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html
- WHO obesity overview: https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight
- AHA sodium: https://www.heart.org/en/healthy-living/healthy-eating/eat-smart/sodium/how-much-sodium-should-i-eat-per-day
- NHANES BPXO_L doc (2021-2023): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.htm
- NHANES BP average notes (1999-2000): https://wwwn.cdc.gov/nchs/data/nhanes/datafiles/bpxdoc.pdf
- NHANES GHB_L (phlebotomy weight WTPH2YR + bridging A1c): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm
- NHANES DR1TOT_L (DR1DRSTZ recall status): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1TOT_L.htm
- NHANES PAQ_L (PAD680 missing codes): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PAQ_L.htm
- NHANES Quality Analyses Guidelines: https://wwwn.cdc.gov/nchs/nhanes/QualityAnalysesGuidelines.aspx
- NHANES Brief Overview 2021-2023 (combinar ciclos / phlebotomy): https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/overviewbrief.aspx?Cycle=2021-2023
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd


# =========================
# Thresholds (guidelines/authorities)
# =========================

THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "htn_accaha_2017": {
        "normal_sbp_lt": 120,
        "normal_dbp_lt": 80,
        "elevated_sbp_ge": 120,
        "elevated_sbp_lt": 130,
        "stage1_sbp_ge": 130,
        "stage1_sbp_lt": 140,
        "stage1_dbp_ge": 80,
        "stage1_dbp_lt": 90,
        "stage2_sbp_ge": 140,
        "stage2_dbp_ge": 90,
    },
    "htn_crisis_safety": {
        "crisis_sbp_ge": 180,
        "crisis_dbp_ge": 120,
    },
    "diabetes_a1c_ada": {
        "normal_a1c_lt": 5.7,
        "prediabetes_a1c_ge": 5.7,
        "prediabetes_a1c_lt": 6.5,
        "diabetes_a1c_ge": 6.5,
        "severe_hyperglycemia_a1c_gt": 10.0,  # operacional (severidade)
    },
    "bmi_adult_cdc": {
        "underweight_lt": 18.5,
        "healthy_ge": 18.5,
        "overweight_ge": 25.0,
        "obesity1_ge": 30.0,
        "obesity2_ge": 35.0,
        "obesity3_ge": 40.0,
        "adult_age_ge": 20,  # CDC adult BMI categories
    },
    "sodium_aha_mg_day": {
        "recommended_max": 2300,
        "ideal_max": 1500,
    },
    "sedentary_minutes_day": {
        "high_ge": 600,  # operacional (~10h/dia)
    },
    "alcohol": {
        "grams_per_standard_drink_us": 14.0,
        "max_drinks_men": 2,
        "max_drinks_women": 1,
    },
    "lipids_general": {
        # Total cholesterol (mg/dL): <200 ideal; 200-239 borderline; >=240 high
        "tc_ideal_lt_mgdl": 200,
        "tc_borderline_ge_mgdl": 200,
        "tc_high_ge_mgdl": 240,
        # HDL low (mg/dL): <40 men, <50 women (metabolic syndrome conventions)
        "hdl_low_men_lt_mgdl": 40,
        "hdl_low_women_lt_mgdl": 50,
    },
    "central_obesity_atp3": {
        # Metabolic syndrome convention (NCEP ATP III / AHA-NHLBI): men >=102 cm, women >=88 cm
        "waist_men_ge_cm": 102,
        "waist_women_ge_cm": 88,
    },
}

QUESTIONNAIRE_MISSING_CODES = {7, 9, 77, 99, 777, 999, 7777, 9999}
DESIGN_VARS = ["SDMVPSU", "SDMVSTRA"]
WEIGHT_BASE_NAMES = ["WTMEC2YR", "WTINT2YR", "WTDRD1", "WTDR2D", "WTPH2YR"]


# =========================
# Logging
# =========================

def setup_logging(out_dir: str, log_level: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("nhanes_step1")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(out_dir, "step1.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# =========================
# IO
# =========================

def load_master(path: str, logger: logging.Logger) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input não encontrado: {path}")

    ext = os.path.splitext(path)[1].lower()
    logger.info(f"A carregar: {path}")

    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError("Formato não suportado. Use .parquet ou .csv")

    logger.info(f"Input: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def save_outputs(df: pd.DataFrame, out_dir: str, base_name: str, logger: logging.Logger) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "parquet": os.path.join(out_dir, f"{base_name}.parquet"),
        "csv": os.path.join(out_dir, f"{base_name}.csv"),
    }
    df.to_parquet(paths["parquet"], index=False)
    df.to_csv(paths["csv"], index=False)
    logger.info(f"Guardado: {paths['parquet']}")
    logger.info(f"Guardado: {paths['csv']}")
    return paths


# =========================
# Utilities
# =========================

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def recode_missing_codes(df: pd.DataFrame, cols: List[str], logger: logging.Logger) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            before = df[c].isna().mean()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c].isin(QUESTIONNAIRE_MISSING_CODES), c] = np.nan
            after = df[c].isna().mean()
            if after > before + 1e-6:
                logger.info(f"Missing recode em {c}: {before:.3f} -> {after:.3f}")
    return df


def unify_duplicate_columns(df, base_names, logger):
    df = df.copy()
    dup_pat = re.compile(r"^(?P<base>.+)_(x|y|\d+)$")  # true merge suffixes

    for base in base_names:
        candidates = [c for c in df.columns if c == base]

        # add only true merge duplicates like WTPH2YR_x, WTPH2YR_y, WTPH2YR_1
        for c in df.columns:
            m = dup_pat.match(c)
            if m and m.group("base") == base:
                candidates.append(c)

        candidates = list(dict.fromkeys(candidates))  # stable unique
        if len(candidates) <= 1:
            continue

        logger.warning(f"Consolidar duplicadas: {base} -> {candidates}")
        consolidated = None
        for c in candidates:
            s = df[c]
            consolidated = s if consolidated is None else consolidated.combine_first(s)
        df[base] = consolidated
        for c in candidates:
            if c != base:
                df.drop(columns=[c], inplace=True, errors="ignore")

    return df

def ensure_cycle_and_keys(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df = df.copy()

    if "Cycle" not in df.columns:
        logger.warning("Coluna 'Cycle' ausente. A criar Cycle='unknown'.")
        df["Cycle"] = "unknown"

    if "SEQN" not in df.columns:
        raise ValueError("Falta coluna obrigatória: SEQN")

    # Chave estável para multi-ciclo
    df["ParticipantKey"] = df["Cycle"].astype(str) + "_" + df["SEQN"].astype(str)

    return df


def sanity_clip_ranges(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Coloca NaN em valores fora de intervalos plausíveis (não “winsoriza”; apenas invalida extremos).
    """
    df = df.copy()
    report: Dict[str, int] = {}

    def invalidate(col: str, lo: float, hi: float) -> None:
        if col not in df.columns:
            return
        s = pd.to_numeric(df[col], errors="coerce")
        bad = (s < lo) | (s > hi)
        n_bad = int(bad.sum())
        if n_bad > 0:
            df.loc[bad, col] = np.nan
            report[col] = n_bad

    # HbA1c %
    invalidate("LBXGH", 2.0, 20.0)
    # BMI
    invalidate("BMXBMI", 10.0, 90.0)
    # Dieta e atividade
    invalidate("DR1TKCAL", 0.0, 20000.0)
    invalidate("DR1TSODI", 0.0, 30000.0)
    invalidate("PAD680", 0.0, 1440.0)

    # BP leituras (serão harmonizadas antes/ao longo do pipeline)
    for i in (1, 2, 3, 4):
        invalidate(f"BPXSY{i}", 40.0, 300.0)
        invalidate(f"BPXDI{i}", 0.0, 200.0)

    if report:
        logger.warning(f"Sanity ranges: invalidados valores fora de gama em {len(report)} colunas.")
    return df, report


# =========================
# Validation
# =========================

@dataclass
class ValidationResult:
    ok: bool
    missing_hard: List[str]
    warnings: List[str]
    presence: Dict[str, bool]


def validate_required_columns(df: pd.DataFrame) -> ValidationResult:
    """
    Hard:
      - SEQN
      - LBXGH
      - BMXBMI
      - BP: BPXSY1/BPXDI1 ou BPXOSY1/BPXODI1 (depois harmonizamos)
    Soft:
      - DR1TSODI, DR1TKCAL (diet)
      - PAD680, PAQ650, PAQ670 (activity)
      - SMQ020, SMQ040 (smoking)
      - DR1DRSTZ (diet recall quality)
      - pesos e desenho: WTMEC2YR/SDMVPSU/SDMVSTRA
    """
    warnings: List[str] = []
    presence: Dict[str, bool] = {}

    hard = ["SEQN", "LBXGH", "BMXBMI"]
    missing_hard = [c for c in hard if c not in df.columns]

    has_bpx = ("BPXSY1" in df.columns) or ("BPXDI1" in df.columns)
    has_bpxo = ("BPXOSY1" in df.columns) or ("BPXODI1" in df.columns)
    if not (has_bpx or has_bpxo):
        missing_hard.append("BPXSY*/BPXDI* or BPXOSY*/BPXODI*")

    soft = ["DR1TSODI", "DR1TKCAL", "PAD680", "PAQ650", "PAQ670", "SMQ020", "SMQ040", "DR1DRSTZ", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]
    for c in soft:
        if c not in df.columns:
            warnings.append(f"Coluna soft ausente: {c}")

    for c in hard + soft:
        presence[c] = c in df.columns
    presence["BP_any"] = has_bpx or has_bpxo

    return ValidationResult(ok=(len(missing_hard) == 0), missing_hard=missing_hard, warnings=warnings, presence=presence)


# =========================
# BP harmonization + NHANES-style means
# =========================

def harmonize_bp(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Normaliza BPXO -> BPX, garantindo BPXSY1..4 e BPXDI1..4.
    No ciclo 2021-2023, existem tipicamente 3 leituras (1..3).
    """
    df = df.copy()

    rename_map = {}
    for i in (1, 2, 3, 4):
        if f"BPXOSY{i}" in df.columns and f"BPXSY{i}" not in df.columns:
            rename_map[f"BPXOSY{i}"] = f"BPXSY{i}"
        if f"BPXODI{i}" in df.columns and f"BPXDI{i}" not in df.columns:
            rename_map[f"BPXODI{i}"] = f"BPXDI{i}"

    if rename_map:
        logger.warning(f"BP harmonize (BPXO->BPX): {rename_map}")
        df.rename(columns=rename_map, inplace=True)

    for i in (1, 2, 3, 4):
        if f"BPXSY{i}" not in df.columns:
            df[f"BPXSY{i}"] = np.nan
        if f"BPXDI{i}" not in df.columns:
            df[f"BPXDI{i}"] = np.nan

    return df


def infer_bp_method(df: pd.DataFrame) -> pd.Series:
    """
    Inferência simples:
    - Se houver coluna BP_method, usar.
    - Caso contrário, usar presença de BPXO (antes de harmonização) ou heurística:
      se BPXSY4 for quase sempre NaN e houver só 3 leituras, pode ser oscilométrico.
    """
    if "BP_method" in df.columns:
        return df["BP_method"].astype(str)

    # Se o pipeline já removeu BPXO_* por rename, inferimos pelo padrão de leituras
    rbp4_missing = df["BPXSY4"].isna().mean() if "BPXSY4" in df.columns else 1.0
    if rbp4_missing > 0.98:
        return pd.Series(["oscillometric_or_3_readings"] * len(df), index=df.index)
    return pd.Series(["auscultatory_or_4_readings"] * len(df), index=df.index)


def nhanes_bp_mean_excl_first(df: pd.DataFrame, sbp_cols: List[str], dbp_cols: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Estilo NHANES (baseado nas notas analíticas clássicas do NHANES):
    - Se existir apenas 1 leitura: média = essa leitura
    - Se existirem >=2 leituras: excluir a 1ª leitura e fazer média das restantes disponíveis
    - DBP: se existirem leituras 0 e também leituras >0, excluir as leituras 0 do cálculo

    Retorna:
    - SBP_mean_nhanes
    - DBP_mean_nhanes
    - BP_n_used (mínimo de leituras usadas entre SBP/DBP; útil para QC)
    """
    sbp = df[sbp_cols].copy()
    dbp = df[dbp_cols].copy()

    sbp = sbp.apply(pd.to_numeric, errors="coerce")
    dbp = dbp.apply(pd.to_numeric, errors="coerce")

    sbp_n = sbp.notna().sum(axis=1)
    dbp_n = dbp.notna().sum(axis=1)

    # SBP
    sbp_use = sbp.copy()
    mask_multi_sbp = sbp_n > 1
    if sbp_cols:
        sbp_use.loc[mask_multi_sbp, sbp_cols[0]] = np.nan
    sbp_mean = sbp_use.mean(axis=1, skipna=True)
    # Se tinha só 1 leitura, manter a 1ª
    sbp_mean = sbp_mean.where(~(sbp_n == 1), sbp[sbp_cols[0]])

    # DBP
    dbp_use = dbp.copy()
    mask_multi_dbp = dbp_n > 1
    if dbp_cols:
        dbp_use.loc[mask_multi_dbp, dbp_cols[0]] = np.nan

    # regra do zero: se existe algum valor >0, zeros passam a NaN
    nonzero_present = (dbp_use.fillna(0) > 0).any(axis=1)
    for c in dbp_cols:
        dbp_use.loc[nonzero_present & (dbp_use[c] == 0), c] = np.nan

    dbp_mean = dbp_use.mean(axis=1, skipna=True)
    dbp_mean = dbp_mean.where(~(dbp_n == 1), dbp[dbp_cols[0]])

    # nº de leituras usadas (aproximação)
    sbp_used_n = sbp_use.notna().sum(axis=1)
    dbp_used_n = dbp_use.notna().sum(axis=1)
    n_used = np.minimum(sbp_used_n, dbp_used_n).astype("float")

    return sbp_mean, dbp_mean, n_used


def compute_bp_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df = df.copy()

    sbp_cols = [f"BPXSY{i}" for i in (1, 2, 3, 4)]
    dbp_cols = [f"BPXDI{i}" for i in (1, 2, 3, 4)]
    df = coerce_numeric(df, sbp_cols + dbp_cols)

    # média simples (para comparação)
    df["SBP_mean_simple"] = df[sbp_cols].mean(axis=1, skipna=True)
    df["DBP_mean_simple"] = df[dbp_cols].mean(axis=1, skipna=True)

    # média NHANES-style
    sbp_mean_nh, dbp_mean_nh, n_used = nhanes_bp_mean_excl_first(df, sbp_cols, dbp_cols)
    df["SBP_mean"] = sbp_mean_nh
    df["DBP_mean"] = dbp_mean_nh
    df["BP_n_used"] = n_used

    df["BP_method_inferred"] = infer_bp_method(df)

    logger.info("BP: criadas SBP_mean/DBP_mean (NHANES-style) + médias simples + BP_n_used + BP_method_inferred.")
    return df


# =========================
# Clinical labels
# =========================

def label_hypertension_accaha_2017(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sbp = df["SBP_mean"]
    dbp = df["DBP_mean"]

    # elegibilidade (guideline adulto): usar 18+ (ACC/AHA é guideline de adultos)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce") if "RIDAGEYR" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    adult18 = age >= 18
    df["Adult18plus_flag"] = adult18.astype("float")

    # Categorias ACC/AHA 2017
    normal = (sbp < THRESHOLDS["htn_accaha_2017"]["normal_sbp_lt"]) & (dbp < THRESHOLDS["htn_accaha_2017"]["normal_dbp_lt"])
    elevated = (sbp >= THRESHOLDS["htn_accaha_2017"]["elevated_sbp_ge"]) & (sbp < THRESHOLDS["htn_accaha_2017"]["elevated_sbp_lt"]) & (dbp < THRESHOLDS["htn_accaha_2017"]["normal_dbp_lt"])
    stage1 = (sbp >= THRESHOLDS["htn_accaha_2017"]["stage1_sbp_ge"]) | (dbp >= THRESHOLDS["htn_accaha_2017"]["stage1_dbp_ge"])
    stage2 = (sbp >= THRESHOLDS["htn_accaha_2017"]["stage2_sbp_ge"]) | (dbp >= THRESHOLDS["htn_accaha_2017"]["stage2_dbp_ge"])

    # Hierarquia: stage2 > stage1 > elevated > normal
    cat = np.select(
        [stage2, stage1, elevated, normal],
        ["Stage2", "Stage1", "Elevated", "Normal"],
        default=None,
    )
    df["HTN_category"] = pd.Series(cat, index=df.index, dtype="object").where(adult18, np.nan)
    # stage code: Normal=0, Elevated=1, Stage1=2, Stage2=3
    stage_code = np.select([stage2, stage1, elevated, normal], [3, 2, 1, 0], default=np.nan)
    df["HTN_stage"] = pd.Series(stage_code, index=df.index).where(adult18, np.nan)

    df["Hypertension"] = np.where(df["HTN_stage"].isna(), np.nan, (df["HTN_stage"] >= 2).astype(int))

    # Safety flag: crise hipertensiva (>=180/120) — aplicável a todas as idades como "alerta"
    crisis = (sbp >= THRESHOLDS["htn_crisis_safety"]["crisis_sbp_ge"]) | (dbp >= THRESHOLDS["htn_crisis_safety"]["crisis_dbp_ge"])
    df["Hypertensive_crisis_flag"] = np.where(sbp.isna() & dbp.isna(), np.nan, crisis.astype(int))

    return df


def label_diabetes_a1c(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LBXGH"] = pd.to_numeric(df["LBXGH"], errors="coerce")
    a1c = df["LBXGH"]

    normal = a1c < THRESHOLDS["diabetes_a1c_ada"]["normal_a1c_lt"]
    pre = (a1c >= THRESHOLDS["diabetes_a1c_ada"]["prediabetes_a1c_ge"]) & (a1c < THRESHOLDS["diabetes_a1c_ada"]["prediabetes_a1c_lt"])
    diab = a1c >= THRESHOLDS["diabetes_a1c_ada"]["diabetes_a1c_ge"]

    df["Diabetes_category"] = pd.Series(
        np.select([diab, pre, normal], ["Diabetes", "Prediabetes", "Normal"], default=None),
        index=df.index,
        dtype="object",
    )
    df["Diabetes"] = np.where(a1c.isna(), np.nan, diab.astype(int))
    df["Prediabetes"] = np.where(a1c.isna(), np.nan, pre.astype(int))

    df["Severe_hyperglycemia_flag"] = np.where(
        a1c.isna(), np.nan, (a1c > THRESHOLDS["diabetes_a1c_ada"]["severe_hyperglycemia_a1c_gt"]).astype(int)
    )
    return df


def label_bmi_adult(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMXBMI"] = pd.to_numeric(df["BMXBMI"], errors="coerce")
    bmi = df["BMXBMI"]

    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce") if "RIDAGEYR" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    adult20 = age >= THRESHOLDS["bmi_adult_cdc"]["adult_age_ge"]
    df["Adult20plus_flag"] = adult20.astype("float")

    under = bmi < THRESHOLDS["bmi_adult_cdc"]["underweight_lt"]
    healthy = (bmi >= THRESHOLDS["bmi_adult_cdc"]["healthy_ge"]) & (bmi < THRESHOLDS["bmi_adult_cdc"]["overweight_ge"])
    over = (bmi >= THRESHOLDS["bmi_adult_cdc"]["overweight_ge"]) & (bmi < THRESHOLDS["bmi_adult_cdc"]["obesity1_ge"])
    ob1 = (bmi >= THRESHOLDS["bmi_adult_cdc"]["obesity1_ge"]) & (bmi < THRESHOLDS["bmi_adult_cdc"]["obesity2_ge"])
    ob2 = (bmi >= THRESHOLDS["bmi_adult_cdc"]["obesity2_ge"]) & (bmi < THRESHOLDS["bmi_adult_cdc"]["obesity3_ge"])
    ob3 = bmi >= THRESHOLDS["bmi_adult_cdc"]["obesity3_ge"]

    cat = np.select(
        [ob3, ob2, ob1, over, healthy, under],
        ["Obesity_Class_3", "Obesity_Class_2", "Obesity_Class_1", "Overweight", "Healthy", "Underweight"],
        default=None
    )
    df["BMI_category"] = pd.Series(cat, index=df.index, dtype="object").where(adult20, np.nan)
    df["Severe_obesity_flag"] = np.where(bmi.isna(), np.nan, (bmi >= THRESHOLDS["bmi_adult_cdc"]["obesity3_ge"]).astype(int))
    return df


# =========================
# Smoking
# =========================

def build_smoking_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    NHANES: SMQ020 (>=100 cigarros na vida) + SMQ040 (fumador atual)
    """
    df = df.copy()
    for c in ["SMQ020", "SMQ040"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    smq020 = df["SMQ020"] if "SMQ020" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    smq040 = df["SMQ040"] if "SMQ040" in df.columns else pd.Series([np.nan] * len(df), index=df.index)

    status = pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")
    status.loc[smq020 == 2] = "Never"

    ever = smq020 == 1
    status.loc[ever & (smq040.isin([1, 2]))] = "Current"
    status.loc[ever & (smq040 == 3)] = "Former"

    df["Smoking_status"] = status
    # Current_smoker_flag: NaN se Unknown
    df["Current_smoker_flag"] = np.where(
        df["Smoking_status"] == "Unknown", np.nan, (df["Smoking_status"] == "Current").astype(int)
    )
    return df


# =========================
# Diet, PA, Lipids, Anthropometry derived features
# =========================

def diet_recall_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    DR1DRSTZ:
      1 = Reliable
      2 = Not reliable / incomplete
    """
    df = df.copy()
    if "DR1DRSTZ" not in df.columns:
        df["DietDay1_recall_status"] = np.nan
        df["DietDay1_reliable_flag"] = np.nan
        return df

    df["DR1DRSTZ"] = pd.to_numeric(df["DR1DRSTZ"], errors="coerce")
    df["DietDay1_recall_status"] = df["DR1DRSTZ"]
    df["DietDay1_reliable_flag"] = np.select(
        [df["DR1DRSTZ"] == 1, df["DR1DRSTZ"] == 2],
        [1, 0],
        default=np.nan
    )
    return df


def lifestyle_and_diet_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PA vars (PAD680 minutes/day sedentary; PAQ650/670 are yes/no)
    if "PAD680" in df.columns:
        df["Sedentary_high_flag"] = np.where(df["PAD680"].isna(), np.nan, (df["PAD680"] >= THRESHOLDS["sedentary_minutes_day"]["high_ge"]).astype(int))
    else:
        df["Sedentary_high_flag"] = np.nan

    if "PAQ650" in df.columns:
        df["Vigorous_activity_flag"] = np.select([df["PAQ650"] == 1, df["PAQ650"] == 2], [1, 0], default=np.nan)
    else:
        df["Vigorous_activity_flag"] = np.nan

    if "PAQ670" in df.columns:
        df["Moderate_activity_flag"] = np.select([df["PAQ670"] == 1, df["PAQ670"] == 2], [1, 0], default=np.nan)
    else:
        df["Moderate_activity_flag"] = np.nan

    df["Any_MVPA_flag"] = np.where(
        df["Vigorous_activity_flag"].isna() & df["Moderate_activity_flag"].isna(),
        np.nan,
        ((df["Vigorous_activity_flag"].fillna(0) == 1) | (df["Moderate_activity_flag"].fillna(0) == 1)).astype(int)
    )

    # Sódio (AHA)
    if "DR1TSODI" in df.columns:
        df["Sodium_over_2300_flag"] = np.where(df["DR1TSODI"].isna(), np.nan, (df["DR1TSODI"] > THRESHOLDS["sodium_aha_mg_day"]["recommended_max"]).astype(int))
        df["Sodium_over_1500_flag"] = np.where(df["DR1TSODI"].isna(), np.nan, (df["DR1TSODI"] > THRESHOLDS["sodium_aha_mg_day"]["ideal_max"]).astype(int))
    else:
        df["Sodium_over_2300_flag"] = np.nan
        df["Sodium_over_1500_flag"] = np.nan

    # Densidades dietéticas e rácios (sem “causalidade”; apenas features)
    if "DR1TKCAL" in df.columns and "DR1TSODI" in df.columns:
        df["Sodium_mg_per_1000kcal"] = np.where(df["DR1TKCAL"] > 0, (df["DR1TSODI"] / df["DR1TKCAL"]) * 1000.0, np.nan)

    if "DR1TKCAL" in df.columns and "DR1TFIBE" in df.columns:
        df["Fiber_g_per_1000kcal"] = np.where(df["DR1TKCAL"] > 0, (df["DR1TFIBE"] / df["DR1TKCAL"]) * 1000.0, np.nan)

    if "DR1TSODI" in df.columns and "DR1TPOTA" in df.columns:
        df["Potassium_to_Sodium_ratio"] = np.where(df["DR1TSODI"] > 0, df["DR1TPOTA"] / df["DR1TSODI"], np.nan)

    # Saturated fat % energy (DGA: <10% kcal)
    if "DR1TKCAL" in df.columns and "DR1TSFAT" in df.columns:
        df["SatFat_energy_pct"] = np.where(df["DR1TKCAL"] > 0, (df["DR1TSFAT"] * 9.0 / df["DR1TKCAL"]) * 100.0, np.nan)
        df["SatFat_over_10pct_flag"] = np.where(df["SatFat_energy_pct"].isna(), np.nan, (df["SatFat_energy_pct"] >= 10.0).astype(int))
    else:
        df["SatFat_energy_pct"] = np.nan
        df["SatFat_over_10pct_flag"] = np.nan

    # Álcool: drinks-equivalentes e flag acima do recomendado (ACC/AHA; DGA)
    if "DR1TALCO" in df.columns and "RIAGENDR" in df.columns:
        df["RIAGENDR"] = pd.to_numeric(df["RIAGENDR"], errors="coerce")  # 1=male, 2=female no NHANES
        drinks = df["DR1TALCO"] / THRESHOLDS["alcohol"]["grams_per_standard_drink_us"]
        df["Alcohol_drinks_equiv_day1"] = drinks

        men = df["RIAGENDR"] == 1
        women = df["RIAGENDR"] == 2
        over = np.where(
            drinks.isna(), np.nan,
            np.where(men, (drinks > THRESHOLDS["alcohol"]["max_drinks_men"]).astype(int),
                np.where(women, (drinks > THRESHOLDS["alcohol"]["max_drinks_women"]).astype(int), np.nan))
        )
        df["Alcohol_over_guideline_flag"] = over
    else:
        df["Alcohol_drinks_equiv_day1"] = np.nan
        df["Alcohol_over_guideline_flag"] = np.nan

    return df


def anthropometry_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Waist/hip/height derived if exists
    if "BMXWAIST" in df.columns and "BMXHIP" in df.columns:
        df["Waist_to_hip_ratio"] = np.where(df["BMXHIP"] > 0, df["BMXWAIST"] / df["BMXHIP"], np.nan)

    if "BMXWAIST" in df.columns and "BMXHT" in df.columns:
        # BMXHT em cm; converter para metros para o ratio
        df["Waist_to_height_ratio"] = np.where(df["BMXHT"] > 0, df["BMXWAIST"] / df["BMXHT"], np.nan)

    # Central obesity (ATP III / AHA-NHLBI) using waist circumference
    if "BMXWAIST" in df.columns and "RIAGENDR" in df.columns:
        men = df["RIAGENDR"] == 1
        women = df["RIAGENDR"] == 2
        waist = pd.to_numeric(df["BMXWAIST"], errors="coerce")
        flag = np.where(
            waist.isna(), np.nan,
            np.where(men, (waist >= THRESHOLDS["central_obesity_atp3"]["waist_men_ge_cm"]).astype(int),
                np.where(women, (waist >= THRESHOLDS["central_obesity_atp3"]["waist_women_ge_cm"]).astype(int), np.nan))
        )
        df["Central_obesity_flag_atp3"] = flag
    else:
        df["Central_obesity_flag_atp3"] = np.nan

    return df


def lipids_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usa colunas quando existirem:
    - Total cholesterol: LBXTC (mg/dL) ou LBDTCSI (mmol/L)
    - HDL: LBDHDD (mg/dL) ou LBDHDDSI (mmol/L)
    Cria flags:
    - Total cholesterol high (>=240 mg/dL) e borderline (>=200)
    - HDL low por sexo
    - TC/HDL ratio (quando possível)
    """
    df = df.copy()

    # Preferir mg/dL se existir; senão SI e converter para mg/dL com fator 38.67
    tc_mgdl = None
    if "LBXTC" in df.columns:
        tc_mgdl = pd.to_numeric(df["LBXTC"], errors="coerce")
    elif "LBDTCSI" in df.columns:
        tc_mgdl = pd.to_numeric(df["LBDTCSI"], errors="coerce") * 38.67

    hdl_mgdl = None
    if "LBDHDD" in df.columns:
        hdl_mgdl = pd.to_numeric(df["LBDHDD"], errors="coerce")
    elif "LBDHDDSI" in df.columns:
        hdl_mgdl = pd.to_numeric(df["LBDHDDSI"], errors="coerce") * 38.67

    if tc_mgdl is not None:
        df["TotalChol_mg_dL"] = tc_mgdl
        df["TC_high_flag"] = np.where(tc_mgdl.isna(), np.nan, (tc_mgdl >= THRESHOLDS["lipids_general"]["tc_high_ge_mgdl"]).astype(int))
        df["TC_borderline_flag"] = np.where(tc_mgdl.isna(), np.nan, ((tc_mgdl >= THRESHOLDS["lipids_general"]["tc_borderline_ge_mgdl"]) & (tc_mgdl < THRESHOLDS["lipids_general"]["tc_high_ge_mgdl"])).astype(int))
    else:
        df["TotalChol_mg_dL"] = np.nan
        df["TC_high_flag"] = np.nan
        df["TC_borderline_flag"] = np.nan

    if hdl_mgdl is not None and "RIAGENDR" in df.columns:
        df["HDL_mg_dL"] = hdl_mgdl
        men = df["RIAGENDR"] == 1
        women = df["RIAGENDR"] == 2
        hdl_low = np.where(
            hdl_mgdl.isna(), np.nan,
            np.where(men, (hdl_mgdl < THRESHOLDS["lipids_general"]["hdl_low_men_lt_mgdl"]).astype(int),
                np.where(women, (hdl_mgdl < THRESHOLDS["lipids_general"]["hdl_low_women_lt_mgdl"]).astype(int), np.nan))
        )
        df["HDL_low_flag"] = hdl_low
    else:
        df["HDL_mg_dL"] = np.nan
        df["HDL_low_flag"] = np.nan

    if (tc_mgdl is not None) and (hdl_mgdl is not None):
        df["TC_to_HDL_ratio"] = np.where(hdl_mgdl > 0, tc_mgdl / hdl_mgdl, np.nan)
    else:
        df["TC_to_HDL_ratio"] = np.nan

    return df


# =========================
# Missingness policy + imputation
# =========================

def apply_missingness_policy(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, int]:
    """
    Política pedida:
    - Só remover se SBP_mean E LBXGH estiverem ambos missing.
    """
    df = df.copy()
    mask_drop = df["SBP_mean"].isna() & df["LBXGH"].isna()
    dropped = int(mask_drop.sum())
    df = df.loc[~mask_drop].copy()
    logger.info(f"Missingness policy: removidas {dropped} linhas (SBP_mean e HbA1c ambos missing).")
    return df, dropped


def impute_by_cycle_median(df: pd.DataFrame, cols: List[str], logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Imputação com mediana por Cycle (apenas para variáveis diet/atividade/derivadas específicas),
    mantendo o pipeline simples e reprodutível.
    """
    df = df.copy()
    report: Dict[str, Dict[str, float]] = {}

    if "Cycle" not in df.columns:
        df["Cycle"] = "unknown"

    for cycle, g in df.groupby("Cycle", dropna=False):
        idx = g.index
        report[str(cycle)] = {}
        for c in cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df.loc[idx, c], errors="coerce")
            med = s.median(skipna=True)
            if pd.isna(med):
                continue
            n_missing = int(s.isna().sum())
            if n_missing > 0:
                df.loc[idx, c] = s.fillna(med)
            report[str(cycle)][c] = float(med)

    logger.info("Imputação por mediana (por ciclo) concluída.")
    return df, report


def rescale_multi_cycle_weights(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, float]:
    """
    Cria pesos *_multi dividindo por nº de ciclos (k).
    Mantém pesos originais intactos.
    Nota: combinar 2021-2023 com ciclos anteriores implica cautela por gap 2020-2021.
    """
    df = df.copy()
    cycles = [c for c in df["Cycle"].dropna().unique().tolist() if str(c) != "nan"]
    k = max(1, len(cycles))
    factors: Dict[str, float] = {}

    for w in ["WTMEC2YR", "WTINT2YR", "WTDRD1", "WTDR2D", "WTPH2YR"]:
        if w in df.columns:
            new = f"{w}_multi"
            df[new] = pd.to_numeric(df[w], errors="coerce") / float(k)
            factors[new] = 1.0 / float(k)

    logger.warning(f"Weight rescale: criados *_multi com k={k}. Pesos originais preservados.")
    return factors


# =========================
# Comorbidity + severity score (auditável, heurístico)
# =========================

def build_comorbidity_and_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Comorbidity"] = np.where(
        df["Hypertension"].isna() | df["Diabetes"].isna(),
        np.nan,
        ((df["Hypertension"] == 1) & (df["Diabetes"] == 1)).astype(int)
    )

    # Scores por domínio (simples, auditáveis; não são "risk calculators")
    # HTN severity: Stage2=2, Stage1=1, Elevated=0.5, Normal=0
    htn_score = np.select(
        [df["HTN_category"] == "Stage2", df["HTN_category"] == "Stage1", df["HTN_category"] == "Elevated", df["HTN_category"] == "Normal"],
        [2.0, 1.0, 0.5, 0.0],
        default=np.nan
    )

    # DM severity: Normal=0, Prediabetes=1, Diabetes=2, +1 se HbA1c>10
    dm_score = np.select(
        [df["Diabetes_category"] == "Diabetes", df["Diabetes_category"] == "Prediabetes", df["Diabetes_category"] == "Normal"],
        [2.0, 1.0, 0.0],
        default=np.nan
    )
    dm_score = np.where(pd.isna(dm_score), np.nan, dm_score + np.where(df["Severe_hyperglycemia_flag"] == 1, 1.0, 0.0))

    # Obesity severity: Overweight=1, Obesity1=2, Obesity2=3, Obesity3=4
    obesity_score = np.select(
        [df["BMI_category"] == "Obesity_Class_3", df["BMI_category"] == "Obesity_Class_2", df["BMI_category"] == "Obesity_Class_1", df["BMI_category"] == "Overweight", df["BMI_category"] == "Healthy", df["BMI_category"] == "Underweight"],
        [4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
        default=np.nan
    )

    # Lifestyle score: sodium>2300 + sedentary_high + current_smoker + alcohol_over_guideline + satfat_over_10%
    lifestyle_flags = []
    for c in ["Sodium_over_2300_flag", "Sedentary_high_flag", "Current_smoker_flag", "Alcohol_over_guideline_flag", "SatFat_over_10pct_flag"]:
        if c in df.columns:
            lifestyle_flags.append(c)

    if lifestyle_flags:
        lifestyle_score = None
        for c in lifestyle_flags:
            s = pd.to_numeric(df[c], errors="coerce")
            lifestyle_score = s if lifestyle_score is None else lifestyle_score.add(s, fill_value=0)
        lifestyle_score = lifestyle_score.where(~df[lifestyle_flags].isna().all(axis=1), np.nan)
    else:
        lifestyle_score = pd.Series([np.nan] * len(df), index=df.index)

    df["HTN_severity_score"] = htn_score
    df["DM_severity_score"] = dm_score
    df["Obesity_severity_score"] = obesity_score
    df["Lifestyle_risk_score"] = lifestyle_score

    # Score global (ajustável)
    df["Cardiometabolic_severity_score"] = (
        pd.Series(htn_score).fillna(0)
        + pd.Series(dm_score).fillna(0)
        + pd.Series(obesity_score).fillna(0)
        + pd.Series(lifestyle_score).fillna(0)
    )

    df["Severity_band"] = pd.cut(df["Cardiometabolic_severity_score"], bins=[-0.1, 3, 7, 100], labels=["Low", "Moderate", "High"])
    return df


# =========================
# Feature tiers + associations (usar "todas as variáveis" com metadados)
# =========================

def build_feature_tiers(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Tiers (listas de colunas) para experimentos:
    - minimal: demografia + antropometria simples
    - core: minimal + BP_mean + HbA1c + BMI_category
    - full: core + dieta + atividade + tabaco + lipids + derived ratios
    """
    minimal_candidates = [
        "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3",
        "DMDEDUC2", "INDFMPIR", "INDHHIN2",
        "BMXBMI", "BMXWAIST", "BMXWT", "BMXHT",
    ]
    minimal = [c for c in minimal_candidates if c in df.columns]

    core_candidates = [
        "SBP_mean", "DBP_mean", "BP_n_used", "BP_method_inferred",
        "LBXGH", "Diabetes_category",
        "BMI_category", "Severe_obesity_flag",
        "HTN_category", "Hypertension",
    ]
    core = minimal + [c for c in core_candidates if c in df.columns]

    full_candidates = [
        # Diet raw
        "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TTFAT", "DR1TSFAT", "DR1TFIBE", "DR1TSODI", "DR1TPOTA", "DR1TALCO",
        # Diet derived
        "Sodium_mg_per_1000kcal", "Fiber_g_per_1000kcal", "Potassium_to_Sodium_ratio", "SatFat_energy_pct",
        # Lifestyle flags
        "Sodium_over_2300_flag", "Sodium_over_1500_flag", "Sedentary_high_flag", "Any_MVPA_flag",
        "Alcohol_drinks_equiv_day1", "Alcohol_over_guideline_flag",
        # Smoking
        "Smoking_status", "Current_smoker_flag",
        # Lipids
        "TotalChol_mg_dL", "HDL_mg_dL", "TC_high_flag", "TC_borderline_flag", "HDL_low_flag", "TC_to_HDL_ratio",
        # Anthropometry derived
        "Waist_to_hip_ratio", "Waist_to_height_ratio", "Central_obesity_flag_atp3",
        # Scores
        "HTN_severity_score", "DM_severity_score", "Obesity_severity_score", "Lifestyle_risk_score", "Cardiometabolic_severity_score", "Severity_band",
    ]
    full = core + [c for c in full_candidates if c in df.columns]

    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return {"minimal": uniq(minimal), "core": uniq(core), "full": uniq(full)}


def build_feature_associations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Gera um registo de associações:
      - por feature: domínios, condições, papel
      - por condição: lista de features
      - por domínio: lista de features

    Nota importante:
    - Isto NÃO afirma causalidade. É um mapa de relevância para explicabilidade e recomendação.
    """
    cols = list(df.columns)

    def domain_of(col: str) -> List[str]:
        d = []
        if col.startswith("BPX") or col in ["SBP_mean", "DBP_mean", "HTN_category", "HTN_stage", "Hypertension", "Hypertensive_crisis_flag", "BP_method_inferred", "BP_n_used"]:
            d.append("blood_pressure")
        if col in ["LBXGH", "Diabetes", "Prediabetes", "Diabetes_category", "Severe_hyperglycemia_flag"]:
            d.append("glycemic")
        if col.startswith("BMX") or col.startswith("Waist_") or col in ["BMI_category", "Severe_obesity_flag", "Central_obesity_flag_atp3"]:
            d.append("anthropometry")
        if col.startswith("DR1") or col.startswith("Sodium_") or col.endswith("_per_1000kcal") or col.endswith("_ratio") or col.startswith("SatFat_") or col.startswith("Alcohol_") or col.startswith("DietDay1_"):
            d.append("diet")
        if col.startswith("PAQ") or col.startswith("PAD") or col.endswith("_activity_flag") or col in ["Sedentary_high_flag", "Any_MVPA_flag"]:
            d.append("physical_activity")
        if col.startswith("SMQ") or col.startswith("Smoking_") or col in ["Current_smoker_flag"]:
            d.append("smoking")
        if col.startswith("TotalChol") or col.startswith("HDL") or col.startswith("TC_") or col.endswith("lip") or col in ["TC_to_HDL_ratio", "TC_high_flag", "TC_borderline_flag", "HDL_low_flag"]:
            d.append("lipids")
        if col in DESIGN_VARS or col in WEIGHT_BASE_NAMES or col.endswith("_multi"):
            d.append("survey_design")
        if col.startswith("RID") or col.startswith("DMD") or col.startswith("IND") or col in ["Cycle", "SEQN", "ParticipantKey"]:
            d.append("demographics_ses")
        if col.endswith("_score") or col.endswith("_band"):
            d.append("risk_scores")
        return sorted(list(set(d))) if d else ["other"]

    def conditions_of(col: str, domains: List[str]) -> List[str]:
        cond = []
        if "blood_pressure" in domains:
            cond.append("hypertension")
        if "glycemic" in domains:
            cond.append("diabetes")
        # Comorbidity engine usa união de domínios cardiometabólicos
        if any(x in domains for x in ["blood_pressure", "glycemic", "anthropometry", "lipids", "diet", "physical_activity", "smoking"]):
            cond.append("comorbidity")
        return sorted(list(set(cond)))

    def role_of(col: str) -> str:
        if col in ["Hypertension", "HTN_category", "HTN_stage", "SBP_mean", "DBP_mean"]:
            return "target_or_criterion"
        if col in ["Diabetes", "Prediabetes", "Diabetes_category", "LBXGH"]:
            return "target_or_biomarker"
        if col in ["BMI_category", "BMXBMI", "Severe_obesity_flag"]:
            return "biomarker_or_condition"
        if col in WEIGHT_BASE_NAMES or col in DESIGN_VARS or col.endswith("_multi"):
            return "survey_weight_or_design"
        if col.endswith("_flag"):
            return "risk_modifier_flag"
        if col.endswith("_score") or col.endswith("_band"):
            return "composite_score"
        return "feature"

    per_feature: Dict[str, Any] = {}
    per_condition: Dict[str, List[str]] = {"hypertension": [], "diabetes": [], "comorbidity": []}
    per_domain: Dict[str, List[str]] = {}

    for c in cols:
        dom = domain_of(c)
        cond = conditions_of(c, dom)
        role = role_of(c)
        per_feature[c] = {"domains": dom, "conditions": cond, "role": role}

        for k in cond:
            per_condition.setdefault(k, []).append(c)
        for d in dom:
            per_domain.setdefault(d, []).append(c)

    # ordenar listas para estabilidade
    for k in per_condition:
        per_condition[k] = sorted(list(set(per_condition[k])))
    for k in per_domain:
        per_domain[k] = sorted(list(set(per_domain[k])))

    return {"per_feature": per_feature, "per_condition": per_condition, "per_domain": per_domain}


# =========================
# Reports
# =========================

def missingness_report(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in cols:
        if c in df.columns:
            miss = int(df[c].isna().sum())
            rows.append({"variable": c, "missing_n": miss, "missing_pct": (miss / n) if n else np.nan})
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def weighted_prevalence(df: pd.DataFrame, indicator_col: str, weight_col: str) -> float:
    if indicator_col not in df.columns or weight_col not in df.columns:
        return np.nan
    dd = df[[indicator_col, weight_col]].copy()
    dd[indicator_col] = pd.to_numeric(dd[indicator_col], errors="coerce")
    dd[weight_col] = pd.to_numeric(dd[weight_col], errors="coerce")
    dd = dd.dropna()
    if dd.empty:
        return np.nan
    w = dd[weight_col].astype(float)
    y = dd[indicator_col].astype(float)
    if float(w.sum()) == 0.0:
        return np.nan
    return float((w * y).sum() / w.sum())


def prevalence_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    indicators = ["Hypertension", "Diabetes", "Comorbidity", "Sodium_over_2300_flag", "Sedentary_high_flag", "Current_smoker_flag"]
    indicators = [c for c in indicators if c in df.columns]

    # Unweighted
    unw_rows = []
    for c in indicators:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        unw_rows.append({"indicator": c, "n_nonmissing": int(s.shape[0]), "prevalence_unweighted": float(s.mean())})
    unweighted = pd.DataFrame(unw_rows)

    # Weighted (escolha de peso pela natureza do indicador)
    w_rows = []
    for c in indicators:
        if c == "Hypertension":
            w = "WTMEC2YR"
        elif c in ["Diabetes", "Comorbidity"]:
            w = "WTPH2YR" if "WTPH2YR" in df.columns else "WTMEC2YR"
        elif c == "Sodium_over_2300_flag":
            w = "WTDRD1" if "WTDRD1" in df.columns else "WTMEC2YR"
        else:
            w = "WTMEC2YR"

        prev = weighted_prevalence(df, c, w) if w in df.columns else np.nan
        w_rows.append({"indicator": c, "weight_used": w, "prevalence_weighted": prev})

    weighted = pd.DataFrame(w_rows)
    return unweighted, weighted


# =========================
# Unit tests (synthetic)
# =========================

def run_unit_tests(logger: logging.Logger) -> None:
    """
    Testes sintéticos para:
    - BP mean excluir 1ª leitura quando >=2
    - DBP zeros regra
    - thresholds HTA/DM/BMI
    """
    test = pd.DataFrame({
        "SEQN": [1, 2, 3, 4],
        "Cycle": ["X"] * 4,
        "RIDAGEYR": [45, 45, 45, 16],  # último é <18
        "RIAGENDR": [1, 2, 1, 1],
        # BP: 4 leituras (com 1ª mais alta)
        "BPXSY1": [140, 120, 118, 140],
        "BPXSY2": [130, 119, 118, 130],
        "BPXSY3": [130, 118, np.nan, 130],
        "BPXSY4": [np.nan, np.nan, np.nan, np.nan],
        "BPXDI1": [90, 79, 0, 90],
        "BPXDI2": [85, 79, 70, 85],
        "BPXDI3": [85, 78, 0, 85],
        "BPXDI4": [np.nan, np.nan, np.nan, np.nan],
        "LBXGH": [6.6, 5.8, 5.4, 6.6],
        "BMXBMI": [32, 27, 22, 32],
        "DR1TSODI": [3000, 2000, 1000, 3000],
        "DR1TKCAL": [2000, 2000, 2000, 2000],
        "DR1TSFAT": [25, 15, 10, 25],
        "PAD680": [650, 300, 100, 650],
        "SMQ020": [1, 2, 1, 1],
        "SMQ040": [1, np.nan, 3, 1],
        "PAQ650": [1, 2, 2, 1],
        "PAQ670": [2, 1, 2, 2],
    })

    test = harmonize_bp(test, logger)
    test = compute_bp_features(test, logger)

    # Linha 1: tem 3 leituras SBP (140,130,130); NHANES-style exclui 1ª => mean=130
    assert abs(test.loc[0, "SBP_mean"] - 130.0) < 1e-6, "BP mean (excluir 1ª) falhou"

    # Linha 3: DBP leituras (0,70,0); exclui 1ª => (70,0) e como há um >0, 0 exclui => mean=70
    assert abs(test.loc[2, "DBP_mean"] - 70.0) < 1e-6, "DBP zero-rule falhou"

    test = label_hypertension_accaha_2017(test)
    # linha 0: SBP_mean=130, DBP_mean=85 => stage1 (adulto)
    assert test.loc[0, "HTN_category"] in ["Stage1", "Stage2"], "HTN category esperado >=Stage1"
    # linha 3: idade 16 => HTN_category deve ser NaN (guideline adulto)
    assert pd.isna(test.loc[3, "HTN_category"]), "HTN category devia ser NaN em <18"

    test = label_diabetes_a1c(test)
    assert test.loc[0, "Diabetes"] == 1, "A1c 6.6 devia ser diabetes"
    assert test.loc[1, "Prediabetes"] == 1, "A1c 5.8 devia ser prediabetes"

    test = label_bmi_adult(test)
    assert test.loc[0, "BMI_category"] == "Obesity_Class_1", "BMI 32 devia ser Obesity Class 1"
    assert pd.isna(test.loc[3, "BMI_category"]), "BMI_category devia ser NaN em <20 (CDC adult categories)"

    logger.info("Unit tests: OK.")


# =========================
# Manifest
# =========================

def write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_manifest(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    validation: ValidationResult,
    out_dir: str,
    outputs: Dict[str, str],
    dropped_rows: int,
    impute_report: Dict[str, Dict[str, float]],
    sanity_report: Dict[str, int],
    weight_rescale_factors: Dict[str, float],
    extra_warnings: List[str],
) -> str:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "input_shape": {"rows": int(df_in.shape[0]), "cols": int(df_in.shape[1])},
        "output_shape": {"rows": int(df_out.shape[0]), "cols": int(df_out.shape[1])},
        "cycle_counts": df_out["Cycle"].value_counts(dropna=False).to_dict() if "Cycle" in df_out.columns else {},
        "validation": {
            "ok": validation.ok,
            "missing_hard": validation.missing_hard,
            "warnings": validation.warnings,
            "presence": validation.presence,
        },
        "dropped_rows_missing_both_bp_and_a1c": int(dropped_rows),
        "thresholds": THRESHOLDS,
        "sanity_invalidated_counts": sanity_report,
        "imputation_medians_by_cycle": impute_report,
        "weight_rescale_factors": weight_rescale_factors,
        "outputs": outputs,
        "warnings": extra_warnings,
    }
    path = os.path.join(out_dir, "manifest_step1.json")
    write_json(manifest, path)
    return path


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="NHANES Step 1 — Clinical Target Engineering (production)")
    parser.add_argument("--input", type=str, default="nhanes_out/NHANES_Master_raw.parquet", help="Path para NHANES_Master_raw.parquet ou .csv")
    parser.add_argument("--outdir", type=str, default="nhanes_step1_out", help="Diretório de output")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--rescale-weights", action="store_true", help="Criar pesos *_multi por nº de ciclos (mantém originais)")
    parser.add_argument("--skip-tests", action="store_true", help="Ignorar unit tests sintéticos")
    args = parser.parse_args()

    logger = setup_logging(args.outdir, args.log_level)
    logger.info("=== NHANES Step 1: Clinical Target Engineering ===")

    if not args.skip_tests:
        run_unit_tests(logger)

    # Load + basic consolidation
    df_in = load_master(args.input, logger)
    df_in = unify_duplicate_columns(df_in, WEIGHT_BASE_NAMES, logger)
    df_in = ensure_cycle_and_keys(df_in, logger)

    # Validate
    validation = validate_required_columns(df_in)
    if not validation.ok:
        raise ValueError(f"Validação falhou. Colunas hard em falta: {validation.missing_hard}")

    # Recode missing codes (questionnaire-style) cedo
    df = df_in.copy()
    df = recode_missing_codes(df, ["SMQ020", "SMQ040", "PAD680", "PAQ650", "PAQ670", "DR1DRSTZ"], logger)

    # Harmonize BP + compute features
    df = harmonize_bp(df, logger)
    df = compute_bp_features(df, logger)

    # Sanity ranges
    df, sanity_report = sanity_clip_ranges(df, logger)

    # Labels
    df = label_hypertension_accaha_2017(df)
    df = label_diabetes_a1c(df)
    df = label_bmi_adult(df)

    # Smoking
    df = build_smoking_status(df)

    # Diet quality
    df = diet_recall_quality(df)

    # Lifestyle + diet derived + anthropometry derived + lipids derived
    df = lifestyle_and_diet_features(df)
    df = anthropometry_derived(df)
    df = lipids_derived(df)

    # Missingness policy
    df, dropped = apply_missingness_policy(df, logger)

    # Impute diet/activity (pedido): usar medianas por ciclo
    imputable = []
    # Diet raw
    for c in ["DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TTFAT", "DR1TSFAT", "DR1TFIBE", "DR1TSODI", "DR1TPOTA", "DR1TALCO", "PAD680"]:
        if c in df.columns:
            imputable.append(c)
    # Activity flags (se quiseres que não fiquem NaN quando missing)
    for c in ["Vigorous_activity_flag", "Moderate_activity_flag"]:
        if c in df.columns:
            imputable.append(c)

    df, impute_report = impute_by_cycle_median(df, imputable, logger)

    # Recalcular features dietéticas baseadas em calorias após imputação (para evitar NaN por missing)
    df = lifestyle_and_diet_features(df)

    # Comorbidity + scores
    df = build_comorbidity_and_scores(df)

    # Optional: rescale weights
    weight_factors: Dict[str, float] = {}
    if args.rescale_weights:
        weight_factors = rescale_multi_cycle_weights(df, logger)

    # Feature tiers + associations JSON
    tiers = build_feature_tiers(df)
    associations = build_feature_associations(df)

    os.makedirs(args.outdir, exist_ok=True)
    tiers_path = os.path.join(args.outdir, "feature_tiers.json")
    assoc_path = os.path.join(args.outdir, "feature_associations.json")
    write_json(tiers, tiers_path)
    write_json(associations, assoc_path)

    # Reports
    report_dir = os.path.join(args.outdir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    key_cols = [
        "SBP_mean", "DBP_mean", "BP_n_used",
        "LBXGH", "Diabetes_category",
        "BMXBMI", "BMI_category",
        "DR1TSODI", "DR1TKCAL", "SatFat_energy_pct",
        "PAD680", "Sedentary_high_flag",
        "Hypertension", "Diabetes", "Comorbidity",
        "WTMEC2YR", "WTDRD1", "WTPH2YR", "SDMVPSU", "SDMVSTRA",
    ]
    miss = missingness_report(df, [c for c in key_cols if c in df.columns])
    miss_path = os.path.join(report_dir, "missingness_report.csv")
    miss.to_csv(miss_path, index=False)

    prev_unw, prev_w = prevalence_tables(df)
    prev_unw_path = os.path.join(report_dir, "prevalence_unweighted.csv")
    prev_w_path = os.path.join(report_dir, "prevalence_weighted_point_estimates.csv")
    prev_unw.to_csv(prev_unw_path, index=False)
    prev_w.to_csv(prev_w_path, index=False)

    # Save engineered dataset
    outputs = save_outputs(df, args.outdir, "NHANES_Master_engineered", logger)
    outputs.update({
        "feature_tiers_json": tiers_path,
        "feature_associations_json": assoc_path,
        "missingness_report_csv": miss_path,
        "prevalence_unweighted_csv": prev_unw_path,
        "prevalence_weighted_csv": prev_w_path,
        "log": os.path.join(args.outdir, "step1.log"),
    })

    extra_warnings = [
        "2021–2023: existe gap (abril 2020–jul 2021). Combinar ciclos requer cautela.",
        "2021–2023: usar WTPH2YR para variáveis de analitos sanguíneos (ex.: HbA1c) quando disponível.",
        "Diet recall 2021–2023 Day1 foi por telefone (mudança de modo); acknowledge em análises.",
        "Sedentarismo >=600 min/dia é um cutoff operacional baseado em evidência, não diagnóstico.",
        "Scores (HTN/DM/Lifestyle/Cardiometabolic) são heurísticos e devem ser tratados como features auxiliares.",
    ]

    manifest_path = build_manifest(
        df_in=df_in,
        df_out=df,
        validation=validation,
        out_dir=args.outdir,
        outputs=outputs,
        dropped_rows=dropped,
        impute_report=impute_report,
        sanity_report=sanity_report,
        weight_rescale_factors=weight_factors,
        extra_warnings=extra_warnings,
    )

    logger.info(f"Manifest: {manifest_path}")
    logger.info("=== Step 1 concluído com sucesso ===")


if __name__ == "__main__":
    main()