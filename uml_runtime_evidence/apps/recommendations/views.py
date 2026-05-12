from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

from apps.recommendations.services.live_pipeline import run_live_patient_pipeline
from hybrid_engine.engine import generate_recommendations

from .models import RecommendationRun, RecommendationItem


def _patient_followup_label(priority: int) -> str:
    if priority <= 1:
        return "Soon"
    if priority <= 3:
        return "Within 3 months"
    return "When convenient"


def _patient_priority_label(priority: int) -> str:
    if priority <= 1:
        return "Top recommendation"
    if priority <= 3:
        return "Recommended"
    return "Helpful option"


@login_required
def generate(request):
    if request.method == "POST":
        pipeline_mode = "live_single_patient_pipeline"
        pipeline_notes = ""
        recs = []

        try:
            live_result = run_live_patient_pipeline(request.user)
            recs = live_result.recommendations
            pipeline_notes = (
                "Generated using the full live single-patient pipeline: "
                "rules engine, cold-start safety/context, ranking, ML reranking/fusion, "
                "bundling, and explainability. "
                f"Manifest: {live_result.manifest}"
            )

            messages.success(request, "Your recommendations were generated from your saved health data.")

        except Exception as exc:
            pipeline_mode = "reference_match_fallback"

            fallback_recs = generate_recommendations(
                user_context={"user_id": request.user.id}
            )

            recs = fallback_recs
            pipeline_notes = (
                "Live single-patient pipeline failed and reference-matching fallback was used. "
                f"Failure: {type(exc).__name__}: {exc}"
            )

            messages.warning(
                request,
                "The full live pipeline could not complete, so the system used the safer reference-matching fallback.",
            )

        if not recs:
            messages.warning(
                request,
                "No recommendations were available. Please add health data and try again.",
            )
            return redirect("recs:generate")

        run = RecommendationRun.objects.create(
            user=request.user,
            engine_version=pipeline_mode,
            notes=pipeline_notes,
        )

        for r in recs:
            RecommendationItem.objects.create(
                run=run,
                title=r.title,
                category=r.category,
                priority=r.priority,
                rationale=r.rationale,
                explanation=r.explanation,
                source=r.source[:32],
                confidence=r.confidence,
            )

        return redirect("recs:detail", run_id=run.id)

    return render(request, "recs/generate.html")


@login_required
def history(request):
    runs = (
        RecommendationRun.objects
        .filter(user=request.user)
        .prefetch_related("items")
        .order_by("-created_at")
    )
    return render(request, "recs/history.html", {"runs": runs})


@login_required
def detail(request, run_id: int):
    run = get_object_or_404(
        RecommendationRun.objects.prefetch_related("items"),
        id=run_id,
        user=request.user,
    )

    items = list(run.items.order_by("priority", "id")[:5])

    for item in items:
        item.patient_followup_label = _patient_followup_label(item.priority)
        item.patient_priority_label = _patient_priority_label(item.priority)

    return render(
        request,
        "recs/detail.html",
        {
            "run": run,
            "items": items,
        },
    )
