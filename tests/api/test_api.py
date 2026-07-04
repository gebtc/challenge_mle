"""Integration tests for the FastAPI /predict and /health endpoints."""

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from challenge.api import app

# ---------------------------------------------------------------------------
# Test client
# ---------------------------------------------------------------------------

client = TestClient(app)


class TestBatchPipeline(unittest.TestCase):
    """Test suite for the FastAPI prediction endpoint."""

    # ------------------------------------------------------------------
    # Health endpoint
    # ------------------------------------------------------------------

    def test_health_endpoint(self) -> None:
        """GET /health returns HTTP 200 with ``{"status": "OK"}``."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})

    # ------------------------------------------------------------------
    # Successful prediction
    # ------------------------------------------------------------------

    def test_should_get_predict(self) -> None:
        """POST /predict with valid flight data returns 200 with a
        ``predict`` list of the same length as the input.

        This test checks structure and types rather than exact values
        to avoid fragility from model retraining.
        """
        response = client.post(
            "/predict",
            json={
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas",
                        "TIPOVUELO": "N",
                        "MES": 3,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("predict", body)
        self.assertIsInstance(body["predict"], list)
        self.assertEqual(len(body["predict"]), 1)
        for pred in body["predict"]:
            self.assertIn(pred, [0, 1])

    def test_should_get_predict_known_airline(self) -> None:
        """POST /predict with an airline in TOP_10_FEATURES returns 200
        and a valid prediction."""
        response = client.post(
            "/predict",
            json={
                "flights": [
                    {
                        "OPERA": "Grupo LATAM",
                        "TIPOVUELO": "I",
                        "MES": 7,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("predict", body)
        self.assertEqual(len(body["predict"]), 1)
        self.assertIn(body["predict"][0], [0, 1])

    # ------------------------------------------------------------------
    # Validation errors (should return HTTP 400)
    # ------------------------------------------------------------------

    def test_should_failed_unkown_column_1(self) -> None:
        """MES=13 should return 400."""
        response = client.post(
            "/predict",
            json={
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas",
                        "TIPOVUELO": "N",
                        "MES": 13,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self) -> None:
        """TIPOVUELO='O' should return 400."""
        response = client.post(
            "/predict",
            json={
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas",
                        "TIPOVUELO": "O",
                        "MES": 3,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_3(self) -> None:
        """Unknown airline should return 400."""
        response = client.post(
            "/predict",
            json={
                "flights": [
                    {
                        "OPERA": "Argentinas",
                        "TIPOVUELO": "N",
                        "MES": 3,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_should_return_422_for_invalid_enum(self) -> None:
        """Unknown airline 'Argentinas' returns 400 (validation error)."""
        response = client.post(
            "/predict",
            json={
                "flights": [
                    {
                        "OPERA": "Argentinas",
                        "TIPOVUELO": "N",
                        "MES": 3,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_should_return_422_for_invalid_tipovuelo(self) -> None:
        """Invalid TIPOVUELO 'O' returns 400."""
        response = client.post(
            "/predict",
            json={
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas",
                        "TIPOVUELO": "O",
                        "MES": 3,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
