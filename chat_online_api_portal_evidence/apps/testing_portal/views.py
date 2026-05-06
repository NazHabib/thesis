from django.shortcuts import render
from apps.accounts.permissions import role_required
from apps.accounts.models import User

@role_required(User.Role.ADMIN, User.Role.TESTER)
def dashboard(request):
    return HttpResponse("testing/dashboard.html")

