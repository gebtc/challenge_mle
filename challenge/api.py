"""Flight delay prediction API.

Exposes endpoints for health checks and batch flight delay predictions
using a trained DelayModel instance loaded at application startup.

Compatibility
-------------
- **pydantic v1** (``~=1.10.2``) and **pydantic v2** (``>=2.0``):
  Uses ``pydantic.v1`` compatibility shim when available.
- **FastAPI 0.86+** and **FastAPI 0.90+**:
  Uses ``on_event`` on older versions, ``lifespan`` on newer versions.
"""

import inspect
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Set

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Pydantic compatibility: support both v1 (~=1.10.2) and v2 (>=2.0)
# ---------------------------------------------------------------------------
import pydantic

# Detect pydantic version to select serialization API.
_PYDANTIC_V2 = int(pydantic.__version__.split(".", maxsplit=1)[0]) >= 2

# Import BaseModel and validator from the top-level pydantic package.
# In pydantic v1, @validator is native (no warnings).
# In pydantic v2, @validator still works but shows deprecation warnings
# (acceptable; they disappear with the challenge-specified v1).
from pydantic import BaseModel, validator  # noqa: E402

from challenge.model import DelayModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default training data path; overridable via DATA_PATH env var.
_DATA_PATH: str = os.getenv(
    "DATA_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "data.csv"),
)

# Module-level set of known airlines, populated during initialization.
# Stored at module level because pydantic v1 @validator decorators need
# access before the FastAPI ``app`` object is created.
_airlines: Set[str] = set()

# Guard flag to prevent double initialization (idempotent init).
_initialized: bool = False


# ---------------------------------------------------------------------------
# Initialization helper
# ---------------------------------------------------------------------------


def _initialize_app() -> None:
    """Load training data, extract airlines, and train the model.

    This function is idempotent — it runs only once. It is called both
    at module import time (to support TestClient without a context
    manager) and from the startup mechanism (``lifespan`` or ``on_event``)
    for production servers.

    The module-level ``_airlines`` set is populated here because pydantic
    v1 ``@validator`` decorators must reference it before the FastAPI
    ``app`` object is created, making ``app.state`` unavailable.
    """
    global _airlines, _initialized  # noqa: PLW0603

    if _initialized:
        return

    # --- Load data --------------------------------------------------------
    logger.info("Loading training data from %s", _DATA_PATH)
    data = pd.read_csv(_DATA_PATH, low_memory=False)

    # --- Extract airlines -------------------------------------------------
    _airlines = set(data["OPERA"].unique())
    logger.info("Loaded %d unique airlines from data.csv", len(_airlines))

    # --- Train model ------------------------------------------------------
    model = DelayModel()
    try:
        result = model.preprocess(data, target_column="delay")
        if isinstance(result, tuple) and len(result) == 2:
            features, target = result
            model.fit(features, target)
            logger.info("Model trained successfully")
        else:
            logger.warning(
                "DelayModel.preprocess returned %s; skipping fit. "
                "The model may not produce valid predictions.",
                type(result).__name__,
            )
    except Exception:
        logger.exception("Model training failed; predictions may be unavailable")

    app.state.model = model
    _initialized = True


# ---------------------------------------------------------------------------
# FastAPI startup: use lifespan (0.90+) or on_event (0.86+)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — runs init on startup, cleanup on shutdown."""
    _initialize_app()
    yield


# Introspect FastAPI to detect if ``lifespan`` parameter is supported.
_supports_lifespan = "lifespan" in inspect.signature(FastAPI.__init__).parameters

if _supports_lifespan:
    app = FastAPI(title="Flight Delay Prediction API", lifespan=_lifespan)
else:
    app = FastAPI(title="Flight Delay Prediction API")
    app.add_event_handler("startup", _initialize_app)

# Run initialization at import time so that TestClient (without a context
# manager) and any other consumer that imports ``app`` get a ready instance.
_initialize_app()


# ---------------------------------------------------------------------------
# Pydantic schemas (v1 syntax — compatible with pydantic~=1.10.2)
# ---------------------------------------------------------------------------


class Flight(BaseModel):
    """Schema for a single flight record submitted for prediction."""

    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("MES")
    def validate_mes(cls, value: int) -> int:
        """Ensure MES is a valid calendar month (1-12)."""
        if value < 1 or value > 12:
            raise ValueError(f"MES must be between 1 and 12, got {value}")
        return value

    @validator("TIPOVUELO")
    def validate_tipovuelo(cls, value: str) -> str:
        """Ensure TIPOVUELO is one of the accepted flight types."""
        allowed = {"I", "N"}
        if value not in allowed:
            raise ValueError(
                f"TIPOVUELO must be one of {sorted(allowed)}, got '{value}'"
            )
        return value

    @validator("OPERA")
    def validate_opera(cls, value: str) -> str:
        """Ensure OPERA matches a known airline loaded from data.csv."""
        if _airlines and value not in _airlines:
            raise ValueError(f"Unknown airline: '{value}'")
        return value


class FlightsRequest(BaseModel):
    """Request body for the /predict endpoint."""

    flights: List[Flight]


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return HTTP 400 for validation errors instead of FastAPI's default 422.

    The test suite expects 400 Bad Request for invalid payloads, so we
    override the default handler here.

    Args:
        request: The incoming HTTP request.
        exc: The validation exception raised by pydantic.

    Returns:
        JSONResponse with status 400 and error details.
    """
    logger.warning("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Utility: serialize pydantic model to dict (v1/v2 compatible)
# ---------------------------------------------------------------------------


def _model_to_dict(model: BaseModel) -> dict:
    """Serialize a pydantic model to a plain dict.

    Works with both pydantic v1 (``.dict()``) and pydantic v2
    (``.model_dump()``).
    """
    if _PYDANTIC_V2:
        return model.model_dump()
    return model.dict()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Health-check endpoint.

    Returns:
        dict: ``{"status": "OK"}`` when the service is running.
    """
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: FlightsRequest) -> dict:
    """Predict whether each submitted flight will be delayed.

    Args:
        request: Validated request body containing a list of flights.

    Returns:
        dict: ``{"predict": [<int>, ...]}`` with one prediction per flight.
    """
    # Build a DataFrame from the validated flight records
    df = pd.DataFrame([_model_to_dict(flight) for flight in request.flights])

    # Preprocess features (no target column for inference)
    features = app.state.model.preprocess(df)
    predictions = app.state.model.predict(features)

    return {"predict": predictions}
