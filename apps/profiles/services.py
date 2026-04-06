from apps.accounts.models import User
from .models import ClinicianPatientLink

def clinician_can_access_patient(clinician: User, patient: User) -> bool:
    if clinician.role == User.Role.ADMIN:
        return True
    return ClinicianPatientLink.objects.filter(clinician=clinician, patient=patient).exists()
