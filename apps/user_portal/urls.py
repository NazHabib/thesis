from django.urls import path

from . import views
from . import views_profile
from . import views_alerts

app_name = "user"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("profile/", views_profile.profile, name="profile"),

    path("alerts/", views_alerts.alerts_list, name="alerts"),
    path("alerts/<int:alert_id>/ack/", views_alerts.alert_ack, name="alert_ack"),
]

