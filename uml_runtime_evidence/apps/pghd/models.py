from django.conf import settings
from django.db import models

class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class DatasetFile(models.Model):
    dataset = models.ForeignKey(Dataset, related_name="files", on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to="datasets/")
    original_name = models.CharField(max_length=255, blank=True, default="")

class PipelineRun(models.Model):
    dataset = models.ForeignKey(Dataset, related_name="runs", on_delete=models.CASCADE)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=32, default="created")  # created/running/success/failed
    logs = models.TextField(blank=True, default="")

