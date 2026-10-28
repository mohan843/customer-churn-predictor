"""Customer churn prediction with an interpretable logistic regression model."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

FEATURES = ("tenure_months", "monthly_charges", "support_tickets", "usage_hours")


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


class StandardScaler:
    def fit(self, rows: Sequence[Sequence[float]]) -> "StandardScaler":
        if not rows:
            raise ValueError("At least one row is required")
        self.means = [sum(row[i] for row in rows) / len(rows) for i in range(len(rows[0]))]
        self.scales = []
        for i, mean in enumerate(self.means):
            variance = sum((row[i] - mean) ** 2 for row in rows) / len(rows)
            self.scales.append(math.sqrt(variance) or 1.0)
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales)] for row in rows]


class LogisticChurnModel:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1800) -> None:
        self.learning_rate, self.epochs = learning_rate, epochs

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> "LogisticChurnModel":
        if not rows or len(rows) != len(labels):
            raise ValueError("Rows and labels must be non-empty and equally sized")
        self.weights = [0.0] * (len(rows[0]) + 1)
        for _ in range(self.epochs):
            gradients = [0.0] * len(self.weights)
            for row, label in zip(rows, labels):
                probability = sigmoid(self.weights[0] + sum(w * x for w, x in zip(self.weights[1:], row)))
                error = probability - label
                gradients[0] += error
                for i, value in enumerate(row, 1):
                    gradients[i] += error * value
            for i, gradient in enumerate(gradients):
                self.weights[i] -= self.learning_rate * gradient / len(rows)
        return self

    def probabilities(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [sigmoid(self.weights[0] + sum(w * x for w, x in zip(self.weights[1:], row))) for row in rows]

    def predict(self, rows: Sequence[Sequence[float]], threshold: float = 0.5) -> list[int]:
        return [int(probability >= threshold) for probability in self.probabilities(rows)]


def load_csv(path: str | Path) -> tuple[list[list[float]], list[int]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [[float(row[name]) for name in FEATURES] for row in rows], [int(row["churned"]) for row in rows]


def accuracy(actual: Sequence[int], predicted: Sequence[int]) -> float:
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def train(path: str | Path) -> tuple[LogisticChurnModel, StandardScaler, float]:
    rows, labels = load_csv(path)
    scaler = StandardScaler().fit(rows)
    transformed = scaler.transform(rows)
    model = LogisticChurnModel().fit(transformed, labels)
    return model, scaler, accuracy(labels, model.predict(transformed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate customer churn probability.")
    parser.add_argument("--data", default="data/customers.csv")
    parser.add_argument("--tenure-months", type=float, required=True)
    parser.add_argument("--monthly-charges", type=float, required=True)
    parser.add_argument("--support-tickets", type=float, required=True)
    parser.add_argument("--usage-hours", type=float, required=True)
    args = parser.parse_args()
    model, scaler, score = train(args.data)
    row = [[args.tenure_months, args.monthly_charges, args.support_tickets, args.usage_hours]]
    probability = model.probabilities(scaler.transform(row))[0]
    print(f"Training accuracy: {score:.1%}")
    print(f"Churn probability: {probability:.1%}")
    print("Retention outreach recommended" if probability >= 0.5 else "Customer likely to remain")
