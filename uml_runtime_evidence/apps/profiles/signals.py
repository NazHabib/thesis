from django.db.models.signals import post_save
from django.dispatch import receiver

from apps.accounts.models import User
from .models import PatientProfile


@receiver(post_save, sender=User)
def create_profile(sender, instance: User, created: bool, **kwargs):
    if not created:
        return

    if instance.role != User.Role.ADMIN:
        PatientProfile.objects.get_or_create(user=instance)
