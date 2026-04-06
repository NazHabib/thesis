from django.urls import path
from .views import RankingRunView

urlpatterns = [
    path("run/", RankingRunView.as_view(), name="ranking-run"),
]
