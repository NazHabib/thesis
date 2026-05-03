from django.urls import path

from . import views
from . import views_profile
from . import views_alerts
from . import views_health_data

app_name = "user"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),

    path("profile/", views_profile.profile, name="profile"),
    path("profile/account/", views_profile.account_edit, name="account_edit"),
    path("profile/health/", views_profile.health_edit, name="health_edit"),
    path("profile/delete/", views_profile.account_delete, name="account_delete"),

    path("health-data/", views_health_data.health_data, name="health_data"),
    path("health-data/add/", views_health_data.health_data_add, name="health_data_add"),
    path(
        "health-data/<int:record_id>/",
        views_health_data.health_data_detail,
        name="health_data_detail",
    ),
    path(
        "health-data/<int:record_id>/edit/",
        views_health_data.health_data_edit,
        name="health_data_edit",
    ),
    path(
        "health-data/<int:record_id>/delete/",
        views_health_data.health_data_delete,
        name="health_data_delete",
    ),

    path("alerts/", views_alerts.alerts_list, name="alerts"),
    path("alerts/<int:alert_id>/ack/", views_alerts.alert_ack, name="alert_ack"),
]
