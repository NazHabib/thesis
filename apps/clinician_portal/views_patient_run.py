from pathlib import Path

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from apps.accounts.models import User
from apps.accounts.permissions import role_required
from apps.profiles.services import clinician_can_access_patient
from apps.recommendations.models import RecommendationRun


def _safe_read_jsonl(path: Path) -> pd.DataFrame:
    """
    Read a JSONL artifact safely for thesis/evidence pages.

    Evidence pages should not crash if an artifact is missing, empty,
    or temporarily malformed during development.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.DataFrame()


def _load_thesis_outputs() -> dict[str, pd.DataFrame]:
    base = Path(settings.BASE_DIR)

    return {
        "explained": _safe_read_jsonl(
            base / "apps/ml_layer/output/explained_recommendations.jsonl"
        ),
        "bundles": _safe_read_jsonl(
            base / "apps/ml_layer/output/final_recommendations_after_ml.jsonl"
        ),
        "ranked_before_ml": _safe_read_jsonl(
            base / "apps/ml_layer/output/ranked_before_ml.jsonl"
        ),
        "reranked": _safe_read_jsonl(
            base / "apps/ml_layer/output/reranked_recommendations.jsonl"
        ),
    }


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def thesis_demo_runs(request):
    data = _load_thesis_outputs()
    explained_df = data["explained"]

    if explained_df.empty or "ParticipantKey" not in explained_df.columns:
        participants = []
    else:
        participants = (
            explained_df["ParticipantKey"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

    return render(
        request,
        "clinician/thesis_demo.html",
        {
            "participants": participants,
            "n_participants": len(participants),
            "n_recommendations": len(explained_df),
        },
    )


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def thesis_demo_patient(request, participant_key: str):
    data = _load_thesis_outputs()
    explained_df = data["explained"]
    bundles_df = data["bundles"]
    ranked_before_ml_df = data["ranked_before_ml"]
    reranked_df = data["reranked"]

    if explained_df.empty or "ParticipantKey" not in explained_df.columns:
        recommendations = []
    else:
        patient_df = explained_df[
            explained_df["ParticipantKey"].astype(str) == str(participant_key)
        ].copy()

        if "bundle_rank" in patient_df.columns:
            patient_df = patient_df.sort_values("bundle_rank", kind="stable")
        elif "bundle_score" in patient_df.columns:
            patient_df = patient_df.sort_values(
                "bundle_score",
                ascending=False,
                kind="stable",
            )

        for col in ["fusion_rule_component", "fusion_ml_component"]:
            if col not in patient_df.columns:
                patient_df[col] = 0.0

        rule = pd.to_numeric(
            patient_df["fusion_rule_component"],
            errors="coerce",
        ).fillna(0.0)
        ml = pd.to_numeric(
            patient_df["fusion_ml_component"],
            errors="coerce",
        ).fillna(0.0)

        total = (rule + ml).replace(0, 1)

        patient_df["fusion_rule_pct"] = ((rule / total) * 100).round(1)
        patient_df["fusion_ml_pct"] = ((ml / total) * 100).round(1)

        recommendations = patient_df.to_dict("records")

    if bundles_df.empty or "ParticipantKey" not in bundles_df.columns:
        bundle_rows = []
    else:
        bundle_rows = (
            bundles_df[bundles_df["ParticipantKey"].astype(str) == str(participant_key)]
            .copy()
            .to_dict("records")
        )

    if ranked_before_ml_df.empty or "ParticipantKey" not in ranked_before_ml_df.columns:
        before_candidate_rows = []
    else:
        before_df = ranked_before_ml_df[
            ranked_before_ml_df["ParticipantKey"].astype(str) == str(participant_key)
        ].copy()

        before_sort_col = "total_score" if "total_score" in before_df.columns else None
        if before_sort_col:
            before_df = before_df.sort_values(
                before_sort_col,
                ascending=False,
                kind="stable",
            )

        before_candidate_rows = before_df.head(10).to_dict("records")

    if reranked_df.empty or "ParticipantKey" not in reranked_df.columns:
        after_candidate_rows = []
    else:
        after_df = reranked_df[
            reranked_df["ParticipantKey"].astype(str) == str(participant_key)
        ].copy()

        after_sort_col = "final_score" if "final_score" in after_df.columns else "total_score"
        if after_sort_col in after_df.columns:
            after_df = after_df.sort_values(
                after_sort_col,
                ascending=False,
                kind="stable",
            )

        after_candidate_rows = after_df.head(10).to_dict("records")

    return render(
        request,
        "clinician/thesis_demo_patient.html",
        {
            "participant_key": participant_key,
            "recommendations": recommendations,
            "bundle_rows": bundle_rows,
            "before_candidate_rows": before_candidate_rows,
            "after_candidate_rows": after_candidate_rows,
        },
    )


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def patient_run_detail(request, patient_id: int, run_id: int):
    patient = get_object_or_404(User, id=patient_id)

    if not clinician_can_access_patient(request.user, patient):
        return HttpResponseForbidden("Forbidden: patient not assigned.")

    run = get_object_or_404(RecommendationRun, id=run_id, user=patient)

    return render(
        request,
        "clinician/patient_run_detail.html",
        {
            "patient": patient,
            "run": run,
        },
    )


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def patient_run_review(request, patient_id: int, run_id: int):
    patient = get_object_or_404(User, id=patient_id)

    if not clinician_can_access_patient(request.user, patient):
        return HttpResponseForbidden("Forbidden: patient not assigned.")

    run = get_object_or_404(RecommendationRun, id=run_id, user=patient)

    if request.method != "POST":
        return redirect(
            "clinician:patient_run_detail",
            patient_id=patient.id,
            run_id=run.id,
        )

    run.clinician_reviewed = True
    run.clinician_reviewed_by = request.user
    run.clinician_reviewed_at = timezone.now()
    run.clinician_review_note = request.POST.get("clinician_review_note", "").strip()
    run.save(
        update_fields=[
            "clinician_reviewed",
            "clinician_reviewed_by",
            "clinician_reviewed_at",
            "clinician_review_note",
        ]
    )

    messages.success(request, "Recommendation run marked as clinically reviewed.")
    return redirect(
        "clinician:patient_run_detail",
        patient_id=patient.id,
        run_id=run.id,
    )
