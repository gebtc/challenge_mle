"""Flight Delay Prediction API.

Exposes the FastAPI ``app`` instance as ``application`` for ASGI servers
(e.g., uvicorn, gunicorn).
"""

from challenge.api import app as application

__all__ = ["application"]
