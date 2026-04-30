import csv
import json
from pathlib import Path

from django.conf import settings
from django.shortcuts import render

from apps.accounts.permissions import role_required
from apps.accounts.models import User


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def dashboard(request):
    return render(request, "clinician/dashboard.html")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pipeline_counts(path: Path) -> dict:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return rows[0] if rows else {}


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def system_evidence(request):
    base = Path(settings.BASE_DIR)

    active_manifest = _load_json(
        base / "apps/ml_layer/artifacts/manifests/active_manifest.json"
    )
    pipeline_manifest = _load_json(
        base / "apps/ml_layer/output/full_pipeline_manifest.json"
    )
    pipeline_counts = _load_pipeline_counts(
        base / "apps/ml_layer/output/thesis_tables/pipeline_stage_counts.csv"
    )

    training = active_manifest.get("training", {})
    metrics = active_manifest.get("metrics", {})
    ml_runtime = pipeline_manifest.get("ml_reranking", {})
    cold_start = pipeline_manifest.get("cold_start", {})
    bundling = pipeline_manifest.get("bundling", {})
    explainability = pipeline_manifest.get("explainability", {})

    context = {
        "model": {
            "name": active_manifest.get("model_name"),
            "version": active_manifest.get("model_version"),
            "schema": active_manifest.get("feature_schema_version"),
            "calibration": active_manifest.get("calibration_version"),
            "calibration_method": active_manifest.get("calibration_method"),
            "max_ml_contrib": active_manifest.get("max_ml_contrib"),
            "feature_profile": training.get("feature_profile"),
            "label_scheme": training.get("label_scheme"),
            "min_score_norm": training.get("min_score_norm"),
        },
        "training": training,
        "metrics": metrics,
        "pipeline_counts": pipeline_counts,
        "ml_runtime": ml_runtime,
        "cold_start": cold_start,
        "bundling": bundling,
        "explainability": explainability,
    }

    return render(request, "clinician/system_evidence.html", context)
    
    
@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def model_registry(request):
    base = Path(settings.BASE_DIR)
    manifests_dir = base / "apps/ml_layer/artifacts/manifests"
    active_manifest_path = manifests_dir / "active_manifest.json"

    active_manifest = _load_json(active_manifest_path)
    active_version = active_manifest.get("model_version")

    models = []

    for path in sorted(manifests_dir.glob("manifest_*.json")):
        manifest = _load_json(path)
        training = manifest.get("training", {})
        metrics = manifest.get("metrics", {})

        models.append({
            "file_name": path.name,
            "model_version": manifest.get("model_version", path.stem.replace("manifest_", "")),
            "is_active": manifest.get("model_version") == active_version,
            "model_name": manifest.get("model_name"),
            "feature_profile": training.get("feature_profile"),
            "schema_version": manifest.get("feature_schema_version"),
            "calibration_version": manifest.get("calibration_version"),
            "calibration_method": manifest.get("calibration_method"),
            "train_rows": training.get("train_rows"),
            "val_rows": training.get("val_rows"),
            "test_rows": training.get("test_rows"),
            "temporal_feature_count": training.get("temporal_feature_count", 0),
            "collaborative_feature_count": training.get("collaborative_feature_count", 0),
            "patient_profile_feature_count": training.get("patient_profile_feature_count", 0),
            "label_scheme": training.get("label_scheme"),
            "min_score_norm": training.get("min_score_norm"),
            "max_ml_contrib": manifest.get("max_ml_contrib"),
            "val_roc_auc": metrics.get("val_roc_auc"),
            "val_log_loss": metrics.get("val_log_loss"),
            "val_brier_calibrated": metrics.get("val_brier_calibrated"),
            "val_log_loss_calibrated": metrics.get("val_log_loss_calibrated"),
        })

        models = sorted(models, key=lambda x: str(x["model_version"]))

    prev_auc = None
    for m in models:
        auc = m.get("val_roc_auc")
        if prev_auc is not None and auc is not None:
            m["delta_auc"] = float(auc) - float(prev_auc)
        else:
            m["delta_auc"] = None
        if auc is not None:
            prev_auc = auc

    return render(
        request,
        "clinician/model_registry.html",
        {
            "models": models,
            "active_version": active_version,
            "model_count": len(models),
        },
    )
    
@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def dataset_features(request):
    base = Path(settings.BASE_DIR)

    feature_tiers = _load_json(base / "nhanes_step1_out/feature_tiers.json")
    associations = _load_json(base / "nhanes_step1_out/feature_associations.json")
    active_manifest = _load_json(base / "apps/ml_layer/artifacts/manifests/active_manifest.json")

    per_domain = associations.get("per_domain", {})
    per_feature = associations.get("per_feature", {})
    training = active_manifest.get("training", {})

    def load_csv_rows(path: Path, limit: int = 8) -> list[dict]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))[:limit]

    missingness_rows = load_csv_rows(
        base / "nhanes_step1_out/reports/missingness_report.csv",
        limit=10,
    )
    prevalence_rows = load_csv_rows(
        base / "nhanes_step1_out/reports/prevalence_unweighted.csv",
        limit=10,
    )

    feature_glossary = {
        "RIDAGEYR": "Participant age in years.",
        "RIAGENDR": "Participant sex/gender code.",
        "BMXBMI": "Body mass index.",
        "BMXWAIST": "Waist circumference.",
        "LBXGH": "HbA1c glycaemic biomarker.",
        "SBP_mean": "Mean systolic blood pressure.",
        "DBP_mean": "Mean diastolic blood pressure.",
        "Hypertension": "Derived hypertension indicator.",
        "Diabetes": "Derived diabetes indicator.",
        "Prediabetes": "Derived prediabetes indicator.",
        "DR1TSODI": "Day-1 dietary sodium intake.",
        "DR1TSUGR": "Day-1 total sugar intake.",
        "DR1TFIBE": "Day-1 dietary fibre intake.",
        "Any_MVPA_flag": "Moderate/vigorous physical activity indicator.",
        "Sedentary_high_flag": "High sedentary behaviour flag.",
        "Current_smoker_flag": "Current smoking indicator.",
        "Alcohol_over_guideline_flag": "Alcohol intake above guideline threshold.",
        "TotalChol_mg_dL": "Total cholesterol.",
        "HDL_mg_dL": "HDL cholesterol.",
        "TC_high_flag": "High total cholesterol flag.",
        "HDL_low_flag": "Low HDL cholesterol flag.",
        "HTN_severity_score": "Composite hypertension severity score.",
        "DM_severity_score": "Composite diabetes/glycaemic severity score.",
        "Obesity_severity_score": "Composite obesity severity score.",
        "Lifestyle_risk_score": "Composite lifestyle risk score.",
        "Cardiometabolic_severity_score": "Overall cardiometabolic severity score.",
        "Severity_band": "Categorical severity band used for stratification.",
        "action_code": "Candidate recommendation identifier produced by the rules engine.",
        "action_type": "Type of recommendation action, such as inform, recommend, or escalate.",
        "severity": "Clinical severity level assigned by the rules/ranking layer.",
        "bundle_family": "Clinical family used for grouping related actions into bundles.",
        "confidence_label": "Categorical confidence level assigned to the candidate.",
        "feature_tier": "Available-data tier used by the cold-start and safety layers.",
        "confidence_ceiling_global": "Maximum confidence allowed under the current data/safety context.",
        "coldstart_confidence_ceiling": "Cold-start confidence limit for this candidate.",
        "followup_urgency": "Recommended urgency category for follow-up.",
        "monitoring_need_level": "Estimated monitoring intensity required.",
        "temporal_risk_tier": "Temporal risk category used for follow-up timing.",
        "temporal_data_quality": "Quality of temporal/longitudinal information.",
        "profile_age_group": "Patient profile age category.",
        "profile_cardiometabolic_risk": "Patient profile cardiometabolic risk category.",
        "profile_glycaemic_status": "Patient profile glycaemic status.",
        "profile_bp_status": "Patient profile blood pressure status.",
        "profile_weight_status": "Patient profile weight status.",
        "profile_version": "Version of the patient profile schema.",
        "profile_data_completeness_score": "Completeness score for the patient profile.",
        "profile_current_smoker_flag": "Patient profile smoking flag.",
        "profile_inactive_flag": "Patient profile inactivity flag.",
        "profile_high_alcohol_risk_flag": "Patient profile alcohol-risk flag.",
        "profile_poor_diet_signal_flag": "Patient profile poor-diet signal.",
        "collab_available_flag": "Whether collaborative evidence was available.",
        "collab_neighbor_count": "Number of similar participants used for collaborative support.",
        "collab_support_count": "Number of neighbours supporting this action/family.",
        "collab_support_rate": "Proportion of neighbours supporting the recommendation.",
        "collab_mean_rule_score_norm": "Mean normalized rule score among supporting neighbours.",
        "collab_mean_rank_pre_ml": "Mean pre-ML rank among supporting neighbours.",
        "collab_family_support_rate": "Neighbour support rate for the recommendation family.",
        "collab_weighted_support_rate": "Similarity-weighted collaborative support rate.",
        "collab_weighted_mean_rule_score_norm": "Similarity-weighted mean rule score.",
        "collab_weighted_mean_rank_pre_ml": "Similarity-weighted mean pre-ML rank.",
        "collab_mean_supporting_similarity": "Average similarity of supporting neighbours.",
        "triage_mode": "Safety-first mode indicator for high-risk participants.",
        "bundle_rank_within_participant": "Rank of the candidate within its bundle family for the participant.",
    }

    domain_cards = []
    for domain, features in sorted(per_domain.items()):
        domain_cards.append({
            "domain": domain,
            "count": len(features),
            "features": features[:12],
        })

    glossary_rows = []
    active_features = training.get("feature_columns", [])
    for feature in active_features:
        meta = per_feature.get(feature, {})
        glossary_rows.append({
            "name": feature,
            "description": feature_glossary.get(feature, "Engineered feature used by the rules, ML model, or safety pipeline."),
            "domains": ", ".join(meta.get("domains", [])) if meta else "model/runtime",
            "role": meta.get("role", "model feature") if meta else "model feature",
        })

    context = {
        "minimal_count": len(feature_tiers.get("minimal", [])),
        "core_count": len(feature_tiers.get("core", [])),
        "full_count": len(feature_tiers.get("full", [])),
        "domain_cards": domain_cards,
        "glossary_rows": glossary_rows[:80],
        "total_model_features": len(active_features),
        "temporal_feature_count": training.get("temporal_feature_count"),
        "collaborative_feature_count": training.get("collaborative_feature_count"),
        "patient_profile_feature_count": training.get("patient_profile_feature_count"),
        "missingness_rows": missingness_rows,
        "prevalence_rows": prevalence_rows,
    }

    return render(request, "clinician/dataset_features.html", context)
    
@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def system_behaviour(request):
    base = Path(settings.BASE_DIR)

    pipeline_manifest = _load_json(
        base / "apps/ml_layer/output/full_pipeline_manifest.json"
    )

    cold_start = pipeline_manifest.get("cold_start", {})
    ml_runtime = pipeline_manifest.get("ml_reranking", {})
    bundling = pipeline_manifest.get("bundling", {})
    explainability = pipeline_manifest.get("explainability", {})

    rerank_counts = ml_runtime.get("rerank_status_counts", {})
    fallback_counts = ml_runtime.get("fallback_reason_counts", {})
    feature_tier_counts = cold_start.get("feature_tier_counts", {})
    confidence_ceiling_counts = cold_start.get("confidence_ceiling_global_counts", {})

    total_reranked = sum(rerank_counts.values()) if rerank_counts else 0
    ml_reranked = rerank_counts.get("ml_reranked", 0)
    fallback = rerank_counts.get("rules_only_fallback", 0)

    context = {
        "cold_start": cold_start,
        "ml_runtime": ml_runtime,
        "bundling": bundling,
        "explainability": explainability,
        "feature_tier_counts": feature_tier_counts,
        "confidence_ceiling_counts": confidence_ceiling_counts,
        "rerank_counts": rerank_counts,
        "fallback_counts": fallback_counts,
        "total_reranked": total_reranked,
        "ml_reranked": ml_reranked,
        "fallback": fallback,
        "ml_reranked_pct": round((ml_reranked / total_reranked) * 100, 2) if total_reranked else 0,
        "fallback_pct": round((fallback / total_reranked) * 100, 2) if total_reranked else 0,
    }

    return render(request, "clinician/system_behaviour.html", context)


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def model_performance(request):
    base = Path(settings.BASE_DIR)

    active_manifest = _load_json(
        base / "apps/ml_layer/artifacts/manifests/active_manifest.json"
    )
    pipeline_manifest = _load_json(
        base / "apps/ml_layer/output/full_pipeline_manifest.json"
    )

    metrics = active_manifest.get("metrics", {})
    training = active_manifest.get("training", {})
    ml_runtime = pipeline_manifest.get("ml_reranking", {})
    pipeline_counts = _load_pipeline_counts(
        base / "apps/ml_layer/output/thesis_tables/pipeline_stage_counts.csv"
    )

    context = {
        "model_version": active_manifest.get("model_version"),
        "feature_profile": training.get("feature_profile"),
        "metrics": metrics,
        "training": training,
        "ml_runtime": ml_runtime,
        "pipeline_counts": pipeline_counts,
    }

    return render(request, "clinician/model_performance.html", context)
