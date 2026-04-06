from django.urls import path
from .views import ColdStartRunView

urlpatterns = [
    path('cold-start/run/', ColdStartRunView.as_view(), name='cold-start-run'),
]
