"""Locust stress-test script for the Flight Delay Prediction API.

Usage:
    # Start the API:
    uvicorn challenge:application --host 0.0.0.0 --port 8000

    # Run stress test from another terminal:
    locust -f tests/stress/api_stress.py --host http://127.0.0.1:8000 \\
        --headless --users 100 --spawn-rate 1 --run-time 60s

    # Or via Makefile:
    make stress-test STRESS_URL=http://127.0.0.1:8000
"""

from locust import HttpUser, between, task

# A pool of flight records to simulate realistic traffic.
_FLIGHT_POOL = [
    {"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3},
    {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 7},
    {"OPERA": "Sky Airline", "TIPOVUELO": "N", "MES": 12},
    {"OPERA": "Copa Air", "TIPOVUELO": "I", "MES": 4},
    {"OPERA": "Latin American Wings", "TIPOVUELO": "N", "MES": 10},
    {"OPERA": "American Airlines", "TIPOVUELO": "N", "MES": 1},
    {"OPERA": "Iberia", "TIPOVUELO": "I", "MES": 8},
    {"OPERA": "Air France", "TIPOVUELO": "I", "MES": 5},
]


class ApiUser(HttpUser):
    """Simulated API consumer that sends /predict requests."""

    wait_time = between(0.5, 2.0)  # seconds between tasks

    @task(3)
    def predict(self) -> None:
        """Send a batch prediction request with 1-8 flights."""
        import random

        batch_size = random.randint(1, 8)
        flights = random.sample(_FLIGHT_POOL, min(batch_size, len(_FLIGHT_POOL)))
        self.client.post(
            "/predict",
            json={"flights": flights},
        )

    @task(1)
    def health(self) -> None:
        """Check the health endpoint."""
        self.client.get("/health")
