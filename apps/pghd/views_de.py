from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from apps.accounts.permissions import role_required
from apps.accounts.models import User
from .models import Dataset, DatasetFile, PipelineRun

@role_required(User.Role.ADMIN)
def datasets(request):
    items = Dataset.objects.order_by("-created_at")
    return render(request, "de/datasets.html", {"datasets": items})

@role_required(User.Role.ADMIN)
def dataset_detail(request, dataset_id: int):
    ds = get_object_or_404(Dataset, id=dataset_id)
    return render(request, "de/dataset_detail.html", {"ds": ds})

@role_required(User.Role.ADMIN)
def dataset_create(request):
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        desc = request.POST.get("description", "").strip()
        if not name:
            messages.error(request, "Name is required.")
            return redirect("de:dataset_create")

        ds = Dataset.objects.create(name=name, description=desc, created_by=request.user)
        messages.success(request, "Dataset created.")
        return redirect("de:dataset_detail", dataset_id=ds.id)

    return render(request, "de/dataset_create.html")

@role_required(User.Role.ADMIN)
def dataset_upload(request, dataset_id: int):
    ds = get_object_or_404(Dataset, id=dataset_id)

    if request.method == "POST":
        f = request.FILES.get("file")
        if not f:
            messages.error(request, "Please select a file.")
            return redirect("de:dataset_detail", dataset_id=ds.id)

        DatasetFile.objects.create(dataset=ds, file=f, original_name=f.name)
        messages.success(request, "File uploaded.")
        return redirect("de:dataset_detail", dataset_id=ds.id)

    return redirect("de:dataset_detail", dataset_id=ds.id)

@role_required(User.Role.ADMIN)
def run_profile(request, dataset_id: int):
    ds = get_object_or_404(Dataset, id=dataset_id)

    run = PipelineRun.objects.create(dataset=ds, status="running", logs="Profiling started...\n")
    # Stub: mark as success immediately
    run.logs += "Profiling finished. Basic dataset profiling completed.\n"
    run.status = "success"
    run.finished_at = timezone.now()
    run.save()

    messages.success(request, "Pipeline run completed.")
    return redirect("de:dataset_detail", dataset_id=ds.id)
