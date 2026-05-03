from django.contrib import admin

from .models import RecommendationRun, RecommendationItem


class RecommendationItemInline(admin.TabularInline):
    model = RecommendationItem
    extra = 0
    readonly_fields = (
        "title",
        "category",
        "priority",
        "rationale",
        "explanation",
        "source",
        "confidence",
    )
    can_delete = False


@admin.register(RecommendationRun)
class RecommendationRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "created_at",
        "engine_version",
        "items_count",
        "clinician_reviewed",
        "clinician_reviewed_by",
        "clinician_reviewed_at",
    )
    list_filter = (
        "engine_version",
        "clinician_reviewed",
        "created_at",
    )
    search_fields = (
        "user__username",
        "user__email",
        "engine_version",
        "notes",
        "clinician_review_note",
    )
    readonly_fields = (
        "created_at",
        "notes",
    )
    inlines = [RecommendationItemInline]
    ordering = ("-created_at",)

    def items_count(self, obj):
        return obj.items.count()


@admin.register(RecommendationItem)
class RecommendationItemAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "run",
        "category",
        "priority",
        "source",
        "confidence",
    )
    list_filter = (
        "category",
        "source",
        "priority",
    )
    search_fields = (
        "title",
        "rationale",
        "explanation",
        "run__user__username",
    )
