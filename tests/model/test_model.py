"""Unit tests for the DelayModel class."""

import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from challenge.model import DelayModel, TOP_10_FEATURES

# Path to the training dataset relative to this file.
_DATA_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "data.csv"
)


class TestModel(unittest.TestCase):
    """Test suite for DelayModel preprocessing, fitting, and prediction."""

    def setUp(self) -> None:
        """Load the full dataset once for all tests."""
        self.data = pd.read_csv(_DATA_PATH, low_memory=False)
        self.model = DelayModel(data_path=_DATA_PATH)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def test_model_preprocess_for_training(self) -> None:
        """preprocess() with target_column returns (features, target)
        where features has exactly 10 columns and target is a Series
        named 'delay'."""
        features, target = self.model.preprocess(
            self.data, target_column="delay"
        )
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.Series)
        self.assertEqual(
            features.shape[1],
            10,
            f"Expected 10 features, got {features.shape[1]}",
        )
        self.assertEqual(target.name, "delay")

    def test_model_preprocess_for_serving(self) -> None:
        """preprocess() without target_column returns a DataFrame
        with exactly 10 columns (no target)."""
        # Simulate serving data — only the three input fields
        serving_data = self.data[["OPERA", "TIPOVUELO", "MES"]].head(10)
        features = self.model.preprocess(serving_data)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(
            features.shape[1],
            10,
            f"Expected 10 features, got {features.shape[1]}",
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def test_model_fit(self) -> None:
        """After fit(), the model predicts on a held-out test split
        and meets the expected recall / f1 thresholds from the
        exploratory notebook."""
        features, target = self.model.preprocess(
            self.data, target_column="delay"
        )
        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = train_test_split(  # noqa: N806 — matches sklearn convention
            features, target, test_size=0.33, random_state=42
        )
        self.model.fit(x_train, y_train)
        y_preds = self.model._model.predict(x_test)

        report = classification_report(
            y_test, y_preds, output_dict=True, zero_division=0
        )

        # Thresholds from the DS notebook (model 6.b.iii)
        self.assertGreater(
            report["1"]["recall"],
            0.60,
            f"Recall for class 1 (delays) should be > 0.60, "
            f"got {report['1']['recall']:.3f}",
        )
        self.assertGreater(
            report["1"]["f1-score"],
            0.30,
            f"F1 for class 1 (delays) should be > 0.30, "
            f"got {report['1']['f1-score']:.3f}",
        )
        self.assertLess(
            report["0"]["recall"],
            0.60,
            f"Recall for class 0 should be < 0.60 "
            f"(class imbalance trade-off), got {report['0']['recall']:.3f}",
        )
        self.assertLess(
            report["0"]["f1-score"],
            0.70,
            f"F1 for class 0 should be < 0.70, "
            f"got {report['0']['f1-score']:.3f}",
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def test_model_predict(self) -> None:
        """predict() returns List[int] of the correct length,
        even without an explicit fit() call (lazy auto-training)."""
        serving_data = self.data[["OPERA", "TIPOVUELO", "MES"]].head(10)
        features = self.model.preprocess(serving_data)

        # No explicit fit() — predict() should auto-train
        predictions = self.model.predict(features)

        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 10)
        for pred in predictions:
            self.assertIn(pred, [0, 1])


# Required for standalone unittest runner.
if __name__ == "__main__":
    unittest.main()
