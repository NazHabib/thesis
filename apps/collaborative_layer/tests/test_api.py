# apps/collaborative_layer/tests/test_api.py
from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient


class CollaborativeResolveApiTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="tester",
            password="secret123",
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_resolve_returns_rows(self):
        payload = {
            "participant_key": 1001,
            "use_online_if_missing": True,
            "candidates": [
                {"action_code": "RECOMMEND_DASH", "bundle_family": "DietPattern"},
                {"action_code": "RECOMMEND_BP_RECHECK", "bundle_family": "Monitoring"},
            ],
        }
        response = self.client.post("/api/v1/collab/features/resolve", payload, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("rows", response.data)
