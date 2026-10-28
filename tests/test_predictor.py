import unittest
from pathlib import Path

from predictor import LogisticChurnModel, StandardScaler, accuracy, load_csv, train

DATA = Path(__file__).parents[1] / "data" / "customers.csv"


class ChurnTests(unittest.TestCase):
    def test_dataset_loads(self):
        rows, labels = load_csv(DATA)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(labels), 8)

    def test_scaler_centers_features(self):
        transformed = StandardScaler().fit([[1, 10], [3, 20]]).transform([[1, 10], [3, 20]])
        self.assertAlmostEqual(sum(row[0] for row in transformed), 0.0)

    def test_model_separates_retention_and_churn_examples(self):
        model, scaler, score = train(DATA)
        self.assertGreaterEqual(score, 0.90)
        retained = model.probabilities(scaler.transform([[36, 45, 0, 42]]))[0]
        churned = model.probabilities(scaler.transform([[3, 88, 5, 12]]))[0]
        self.assertLess(retained, 0.5)
        self.assertGreater(churned, 0.5)

    def test_invalid_training_input_rejected(self):
        with self.assertRaises(ValueError):
            LogisticChurnModel().fit([], [])

    def test_accuracy(self):
        self.assertEqual(accuracy([0, 1], [0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
