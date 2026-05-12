"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.
"""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.core.asgi import get_asgi_application
from starlette.applications import Starlette
from starlette.routing import Mount

from apps.api.fastapi_app import fastapi_app

django_app = get_asgi_application()

application = Starlette(
    routes=[
        Mount("/api/fast", app=fastapi_app),
        Mount("/", app=django_app),
    ]
)
