"""Flight delay prediction model.

Implements a Logistic Regression classifier trained on the top 10 features
identified by XGBoost feature importance, with explicit class balancing
to improve recall for the minority (delay) class.

Typical usage::

    model = DelayModel()
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)
    predictions = model.predict(features)

For serving (no target column available)::

    features = model.preprocess(flight_df)
    predictions = model.predict(features)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants — top 10 features selected by XGBoost feature importance
# (see exploration.ipynb, section 5)
# ---------------------------------------------------------------------------

TOP_10_FEATURES: List[str] = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

# Default path to training data relative to this file.
_DEFAULT_DATA_PATH: str = str(
    Path(__file__).resolve().parent.parent / "data" / "data.csv"
)


# ---------------------------------------------------------------------------
# DelayModel
# ---------------------------------------------------------------------------


class DelayModel:
    """Logistic Regression model for flight delay prediction.

    The model uses the top 10 most predictive features (one-hot encoded
    airline, month, and flight type) and applies explicit class weighting
    to handle the imbalanced ``delay`` distribution (~93 % on-time vs ~7 %
    delayed).

    Attributes:
        _model (LogisticRegression | None): Fitted scikit-learn model, or
            ``None`` if ``fit()`` has not been called yet.
        _data_path (str): Path to the training CSV, used for lazy
            auto-training in ``predict()``.
    """

    def __init__(self, data_path: str = _DEFAULT_DATA_PATH) -> None:
        """Initialize an untrained DelayModel.

        Args:
            data_path: Path to the training CSV. Defaults to
                ``<project_root>/data/data.csv``.
        """
        self._model: Optional[LogisticRegression] = None
        self._data_path: str = data_path

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """Transform raw flight records into the top-10 feature matrix.

        The transformation pipeline is:

        1. One-hot encode ``OPERA``, ``TIPOVUELO``, and ``MES``.
        2. Select only the columns listed in :data:`TOP_10_FEATURES`.
        3. Reindex (fill missing columns with ``0``) to guarantee a
           consistent 10-column output regardless of input.
        4. If ``target_column`` is given, compute the ``delay`` target
           from the difference between ``Fecha-O`` and ``Fecha-I``.

        Args:
            data: Raw flight data. For training, must contain at least
                ``OPERA``, ``TIPOVUELO``, ``MES``, ``Fecha-I``, and
                ``Fecha-O``. For serving, only the first three are needed.
            target_column: If ``"delay"`` (or any truthy value), the
                function returns a ``(features, target)`` tuple where
                ``target`` is the ``delay`` column. If ``None``, only
                the feature matrix is returned.

        Returns:
            - ``(pd.DataFrame, pd.Series)`` if ``target_column`` is set.
            - ``pd.DataFrame`` if ``target_column`` is ``None``.

        Raises:
            KeyError: If required columns are missing from ``data``.
        """
        # -- Step 1: One-hot encode categorical variables ---------------
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        # -- Step 2: Select only the top 10 features --------------------
        features = features.reindex(columns=TOP_10_FEATURES, fill_value=0)

        # -- Step 3: If training mode, compute the target ---------------
        if target_column is not None:
            # Compute delay = 1 if min_diff > 15 minutes
            fecha_o = pd.to_datetime(data["Fecha-O"])
            fecha_i = pd.to_datetime(data["Fecha-I"])
            min_diff = (fecha_o - fecha_i).dt.total_seconds() / 60.0
            target = pd.Series(
                np.where(min_diff > 15, 1, 0),
                name="delay",
                index=data.index,
            )
            return features, target

        return features

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        """Train the Logistic Regression model with class balancing.

    The class weights are computed to match the strategy from the
    exploratory notebook (section 6.b.iii)::

        n_y0 = sum(target == 0)   # majority class (on-time)
        n_y1 = sum(target == 1)   # minority class (delay)
        n_total = len(target)

        class_weight = {
            1: n_y0 / n_total,  # up-weight the minority class
            0: n_y1 / n_total,  # down-weight the majority class
        }

        Args:
            features: Training feature matrix (N x 10).
            target: Binary target vector (N,) where 1 = delayed.

        Returns:
            ``None``. The fitted model is stored in ``self._model``.
        """
        # -- Class balancing --------------------------------------------
        n_y0: int = int(sum(target == 0))
        n_y1: int = int(sum(target == 1))
        n_total: int = len(target)

        class_weight = {
            0: n_y1 / n_total,
            1: n_y0 / n_total,
        }

        # -- Train ------------------------------------------------------
        self._model = LogisticRegression(class_weight=class_weight)
        self._model.fit(features, target)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> List[int]:
        """Return binary delay predictions for the given features.

        If the model has not been fitted yet (``self._model is None``),
        this method will **auto-train** using the default dataset. This
        supports the test pattern where ``predict()`` is called without
        an explicit ``fit()``.

        Args:
            features: Feature matrix (N x 10) as returned by
                :meth:`preprocess`.

        Returns:
            List of ``int`` predictions, one per row in ``features``.
            Each element is either ``0`` (on-time) or ``1`` (delayed).
        """
        # Lazy auto-training if needed
        if self._model is None:
            data = pd.read_csv(self._data_path, low_memory=False)
            train_features, target = self.preprocess(
                data, target_column="delay"
            )
            self.fit(train_features, target)

        predictions = self._model.predict(features)
        return predictions.tolist()
