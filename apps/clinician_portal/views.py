from django.http import HttpResponse
from apps.accounts.permissions import role_required
from apps.accounts.models import User

@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def dashboard(request):
    return HttpResponse("clinician/dashboard.html")

