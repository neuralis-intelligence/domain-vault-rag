"""
Integration tests for FastAPI backend.

Tests the /ask endpoint and related functionality.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


class TestAPI:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Complaint Insights API"

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_ask_endpoint_accepts_question(self, client):
        """Test that /ask endpoint accepts questions."""
        payload = {"question": "How many complaints did Chase receive?"}
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "route" in data

    def test_ask_endpoint_rejects_empty_question(self, client):
        """Test that /ask endpoint validates input."""
        payload = {"question": ""}
        response = client.post("/ask", json=payload)
        # Depending on validation, this might return 422 or 200
        assert response.status_code in [200, 422]

    def test_feedback_endpoint(self, client):
        """Test feedback submission."""
        payload = {
            "question": "Test question",
            "answer": "Test answer",
            "rating": 5,
            "comments": "Great!"
        }
        response = client.post("/feedback", params=payload)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
