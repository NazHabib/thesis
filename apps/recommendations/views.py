from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

from hybrid_engine.engine import generate_recommendations
from .models import RecommendationRun, RecommendationItem

@login_required
def generate(request):
    if request.method == "POST":
        run = RecommendationRun.objects.create(user=request.user, engine_version="dev-stub")
        recs = generate_recommendations(user_context={"user_id": request.user.id})

        for r in recs:
            RecommendationItem.objects.create(
                run=run,
                title=r.title,
                category=r.category,
                priority=r.priority,
                rationale=r.rationale,
                explanation=r.explanation,
                source=r.source,
                confidence=r.confidence,
            )

        messages.success(request, "Generated recommendations (stub).")
        return redirect("recs:detail", run_id=run.id)

    return render(request, "recs/generate.html")

@login_required
def history(request):
    runs = RecommendationRun.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "recs/history.html", {"runs": runs})

@login_required
def detail(request, run_id: int):
    run = get_object_or_404(RecommendationRun, id=run_id, user=request.user)
    return render(request, "recs/detail.html", {"run": run})

